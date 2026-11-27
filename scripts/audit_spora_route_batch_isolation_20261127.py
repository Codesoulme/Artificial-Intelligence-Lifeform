"""Compare fixed router features across padding and batch shapes."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import MentionHead, encode, locate, route
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS, make_rows, selected
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-batch-isolation-20261127-v1'
PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
PRIOR = ROOT / 'runs/research/route-disjoint-calibration-20261127-v1/result.json'
PARTNER = '这是一段与所问人物无关的文字，用来观察同批其他句子的长度。' * 5


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
                 'candidate': PARENT / 'candidate.pt', 'prior': PRIOR}
        before = {k: base.sha(v) for k, v in paths.items()}
        previous = json.loads(PRIOR.read_text(encoding='utf8'))
        if any(before[k] != previous['input_hashes'][k] for k in ('base', 'adapter', 'candidate')):
            raise ValueError('Protected source differs from prior experiment')
        payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
        examples = [(label, r) for label, forms in FORMS.items()
                    for r in make_rows(NAMES[16:24], forms) if r['group'] == 'read']
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        stage('loading_model')
        adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        original_side = tokenizer.padding_side
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        locator = MentionHead(model.config.hidden_size).eval(); locator.load_state_dict(payload['locator'])
        intent = IntentHead(model.config.hidden_size).eval(); intent.load_state_dict(payload['intent'])
        bank = IdentityFingerprints(len(NAMES))
        for i, name in enumerate(NAMES): bank.bind(i, name)
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        results = []
        with torch.no_grad():
            for index, (wording, example) in enumerate(examples):
                if index % 4 == 0: stage('encoding_' + str(index))
                sample = {'wording': wording, 'text': example['text'], 'expected_key': example['expected_key'], 'variants': {}}
                reference = None
                for side in ('left', 'right'):
                    tokenizer.padding_side = side
                    for variant in ('single', 'duplicate', 'long_partner'):
                        rows = [example]
                        if variant == 'duplicate': rows = [example, example]
                        if variant == 'long_partner': rows = [example, {'text': PARTNER, 'target': None}]
                        tokens = tokenizer([r['text'] for r in rows], padding=True, add_special_tokens=False, return_tensors='pt')
                        if variant == 'long_partner' and tokens['attention_mask'][0].sum() >= tokens['input_ids'].shape[1]:
                            raise ValueError('Partner did not add padding')
                        data = encode(extractor, rows)
                        hit = locate(locator, data, rows)[0]
                        prob = torch.sigmoid(intent(data[0], data[1]))[0].item()
                        features = data[0][0, :len(example['text'])].float()
                        if reference is None: reference = features.clone()
                        d = (features - reference).abs()
                        result = {'token_width': tokens['input_ids'].shape[1],
                                  'real_tokens': int(tokens['attention_mask'][0].sum()),
                                  'feature_max_abs_delta': float(d.max()), 'feature_mean_abs_delta': float(d.mean()),
                                  'span': hit['name'], 'span_exact': hit['name'] == example['target'],
                                  'margin': hit['margin'], 'intent_probability': prob, 'intent_pass': prob >= .5,
                                  'expected_key': example['expected_key'],
                                  'raw_key': route(bank, example['text'], hit, -1000000.)}
                        result['old_key'] = selected(result, payload['locator_threshold'])
                        result['calibrated_key'] = selected(result, previous['calibrated_cutoff'])
                        sample['variants'][side + '_' + variant] = result
                results.append(sample)
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        output = {'rows': results, 'original_padding_side': original_side, 'input_hashes': before,
                  'input_hashes_after': after, 'seconds': time.perf_counter() - started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
