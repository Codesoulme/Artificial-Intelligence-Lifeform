"""Isolate explicit valid-token positions in frozen routing feature extraction."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import MentionHead, encode, locate, route
from scripts.audit_spora_route_batch_isolation_20261127 import PARTNER
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS, make_rows, selected
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-position-control-20261127-v1'
PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
CAL = ROOT / 'runs/research/route-disjoint-calibration-20261127-v1/result.json'
PRIOR = ROOT / 'runs/research/route-batch-isolation-20261127-v1/result.json'


class ControlledFeatures(FrozenTurnFeatures):
    def __init__(self, *args, explicit_positions=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_positions = explicit_positions

    @torch.no_grad()
    def __call__(self, texts):
        width = max(map(len, texts))
        encoded = self.tokenizer(list(texts), padding=True, add_special_tokens=False,
                                 return_tensors='pt', return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')
        if self.explicit_positions:
            position = encoded['attention_mask'].long().cumsum(-1) - 1
            position.masked_fill_(encoded['attention_mask'] == 0, 1)
            encoded['position_ids'] = position
        positions = torch.zeros(len(texts), width, dtype=torch.long)
        for i, row in enumerate(offsets.tolist()):
            for token, (begin, end) in enumerate(row):
                positions[i, begin:end] = token
        layers = self.model.layers
        self.model.layers = nn.ModuleList(layers[:self.depth])
        try:
            hidden = []
            for begin in range(0, len(texts), 12):
                subset = slice(begin, begin + 12)
                local = {k: v[subset].to(self.device) for k, v in encoded.items()}
                output = self.model(**local, use_cache=False).last_hidden_state
                indices = positions[subset].to(self.device)
                hidden.append(output.gather(1, indices.unsqueeze(-1).expand(-1, -1, output.shape[-1])).cpu())
            return torch.cat(hidden)
        finally:
            self.model.layers = layers


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def stage(name):
        v = {'stage': name, 'seconds': time.perf_counter() - started}
        (OUT / 'stage.json').write_text(json.dumps(v), encoding='utf8')
        print(json.dumps(v), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'candidate': PARENT / 'candidate.pt', 'calibration': CAL, 'prior': PRIOR}
        before = {k: base.sha(v) for k, v in paths.items()}
        previous = json.loads(PRIOR.read_text(encoding='utf8'))
        calibration = json.loads(CAL.read_text(encoding='utf8'))
        if any(before[k] != previous['input_hashes'][k] for k in ('base', 'adapter', 'candidate')):
            raise ValueError('Protected source differs from previous experiment')
        payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
        examples = [(label, r) for label, forms in FORMS.items()
                    for r in make_rows(NAMES[16:24], forms) if r['group'] == 'read']
        if len(examples) != len(previous['rows']): raise ValueError('Prior sample count differs')
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        stage('loading_model')
        adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        locator = MentionHead(model.config.hidden_size).eval(); locator.load_state_dict(payload['locator'])
        intent = IntentHead(model.config.hidden_size).eval(); intent.load_state_dict(payload['intent'])
        bank = IdentityFingerprints(len(NAMES))
        for i, name in enumerate(NAMES): bank.bind(i, name)
        normal = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        explicit = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12, explicit_positions=True)
        results = []
        with torch.no_grad():
            for index, (wording, example) in enumerate(examples):
                if index % 4 == 0: stage('encoding_' + str(index))
                if previous['rows'][index]['text'] != example['text']: raise ValueError('Prior ordering differs')
                sample = {'wording': wording, 'text': example['text'], 'variants': {}}
                reference = None
                for name, side, extractor, paired in [('single', 'left', normal, False),
                    ('left_default', 'left', normal, True), ('left_explicit', 'left', explicit, True),
                    ('right_default', 'right', normal, True)]:
                    tokenizer.padding_side = side
                    rows = [example, {'text': PARTNER, 'target': None}] if paired else [example]
                    data = encode(extractor, rows)
                    hit = locate(locator, data, rows)[0]
                    prob = torch.sigmoid(intent(data[0], data[1]))[0].item()
                    features = data[0][0, :len(example['text'])].float()
                    if reference is None: reference = features.clone()
                    d = (features - reference).abs()
                    item = {'feature_max_abs_delta': float(d.max()), 'feature_mean_abs_delta': float(d.mean()),
                            'span': hit['name'], 'margin': hit['margin'], 'intent_probability': prob,
                            'intent_pass': prob >= .5, 'expected_key': example['expected_key'],
                            'raw_key': route(bank, example['text'], hit, -1000000.)}
                    item['old_key'] = selected(item, payload['locator_threshold'])
                    item['calibrated_key'] = selected(item, calibration['calibrated_cutoff'])
                    sample['variants'][name] = item
                results.append(sample)
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        output = {'rows': results, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
