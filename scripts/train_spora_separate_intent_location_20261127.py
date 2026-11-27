"""Compare frozen old routing to separately trained intent and span heads."""
import json
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import (MentionHead, TRAIN_NAMES, CAL_NAMES,
                                                        encode, locate, route, threshold)
from scripts.audit_spora_route_contrast_20261127 import cases as old_cases
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

PARENT = ROOT / 'runs/spora-isolated-name-routes-v1'
OUT = ROOT / 'runs/research/separate-intent-location-20261127-v2'
SEED, EPOCHS, LR = 26301, 20, .002
FORMS = {
 'train': [
  ('read', '请查一下{a}喜欢喝什么。'), ('read', '你记得{a}最爱哪种饮品吗？'),
  ('excluded', '别查{a}喜欢的饮品，只算二加三。'), ('excluded', '不用回忆{a}的口味，告诉我今天心情如何。'),
  ('word_task', '名字{a}一共有几个字？'), ('word_task', '请把{a}这个名字写出来。'),
  ('unknown_longer', '请查一下{u}喜欢喝什么。'), ('unknown_longer', '你记得{u}最爱哪种饮品吗？'),
  ('two_people', '说说{a}和{b}分别喜欢喝什么。'), ('two_people', '查查{a}、{b}各自的饮品偏好。')],
 'cal': [('read', '帮我找出{a}通常喝什么。'), ('excluded', '不问{a}的喜好，只讲天气。'),
         ('word_task', '请数数{a}这个名字的字数。'), ('unknown_longer', '帮我找出{u}通常喝什么。'),
         ('two_people', '{a}与{b}的饮品偏好各是什么？')],
 'new': [('read', '能查一下{a}常点的饮品吗？'), ('excluded', '别翻{a}的喝茶记录，算算三加四。'),
         ('word_task', '请数一数“{a}”含有几个汉字。'), ('unknown_longer', '能查一下{u}常点的饮品吗？'),
         ('two_people', '帮我分别回忆{a}和{b}常喝哪一种。')],
}


def rows_for(names, forms, is_eval=False):
    result = []
    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        unknown = name + '宁'
        if len(unknown) > 4 or unknown in NAMES:
            raise ValueError('Unknown name invalid or enrolled')
        for group, form in forms:
            target = name if group == 'read' else unknown if group == 'unknown_longer' else None
            result.append({'text': form.format(a=name, b=other, u=unknown), 'group': group,
                           'target': target, 'intent': int(target is not None),
                           'expected_key': NAMES.index(name) if is_eval and group == 'read' else None})
    return result


class IntentHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, features, valid):
        pooled = (F.normalize(features.float(), dim=-1) * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        return self.linear(pooled).squeeze(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'locator': PARENT / 'locator.pt', 'report': PARENT / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        old = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['locator'] != old['locator_sha256']:
            raise ValueError('Original locator mismatch')
        for key in ('base', 'adapter'):
            matches = [v for k, v in old['sources'].items() if Path(k).name == paths[key].name]
            if len(matches) != 1 or matches[0] != before[key]:
                raise ValueError('Original source mismatch: ' + key)
        assert not (set(TRAIN_NAMES) | set(CAL_NAMES)) & set(NAMES)
        train = rows_for(TRAIN_NAMES, FORMS['train'])
        cal = rows_for(CAL_NAMES, FORMS['cal'])
        previous = old_cases()
        for i, row in enumerate(previous):
            name = NAMES[i // 5]
            row['target'] = name if row['group'] == 'read' else name + '宁' if row['group'] == 'unknown_longer' else None
            row['intent'] = int(row['target'] is not None)
            row['expected_key'] = row.pop('expected')
        new = rows_for(NAMES[:8], FORMS['new'], is_eval=True)
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        adapter = torch.load(paths['adapter'], weights_only=True, map_location='cpu')
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        features = {}
        for label, rows in [('train', train), ('cal', cal), ('previous', previous), ('new', new)]:
            features[label] = encode(extractor, rows)
            print(json.dumps({'encoded': label, 'rows': len(rows)}), flush=True)
        head = MentionHead(model.config.hidden_size)
        intent = IntentHead(model.config.hidden_size)
        optimizer = torch.optim.AdamW(list(head.parameters()) + list(intent.parameters()), lr=LR)
        generator = torch.Generator().manual_seed(SEED)
        x, mask, target = features['train']
        labels = torch.tensor([r['intent'] for r in train], dtype=torch.float32)
        trace = []
        for epoch in range(EPOCHS):
            for batch in torch.randperm(len(train), generator=generator).split(32):
                span_loss = F.cross_entropy(head(x[batch], mask[batch]), target[batch])
                intent_loss = F.binary_cross_entropy_with_logits(intent(x[batch], mask[batch]), labels[batch])
                optimizer.zero_grad(set_to_none=True)
                (span_loss + intent_loss).backward()
                optimizer.step()
            if epoch in (4, 9, 19):
                trace.append({'epoch': epoch + 1, 'span_loss': float(span_loss.detach()), 'intent_loss': float(intent_loss.detach())})
        head.eval();intent.eval()
        cutoff = threshold(cal, locate(head, features['cal'], cal))
        previous_head = MentionHead(model.config.hidden_size).eval()
        previous_head.load_state_dict(torch.load(paths['locator'], map_location='cpu', weights_only=True))
        bank = IdentityFingerprints(len(NAMES))
        for i, name in enumerate(NAMES):
            bank.bind(i, name)
        results = []
        with torch.no_grad():
            for split, rows in [('previous', previous), ('new', new)]:
                data = features[split]
                predictions = locate(head, data, rows)
                old_predictions = locate(previous_head, data, rows)
                probabilities = torch.sigmoid(intent(data[0], data[1])).tolist()
                for i, (row, hit, prior, probability) in enumerate(zip(rows, predictions, old_predictions, probabilities, strict=True)):
                    selected = route(bank, row['text'], hit, cutoff) if probability >= .5 else None
                    old_selected = route(bank, row['text'], prior, old['cutoff'])
                    results.append({'set': split, 'index': i, 'group': row['group'], 'intent_expected': row['intent'],
                                    'intent_probability': probability, 'span_correct': hit['name'] == row['target'],
                                    'span_margin': hit['margin'], 'span_above_cutoff': hit['margin'] > cutoff,
                                    'key_without_margin_gate': route(bank, row['text'], hit, float('-inf')),
                                    'new_selected': selected, 'old_selected': old_selected, 'expected_key': row['expected_key']})
        torch.save({'intent': intent.state_dict(), 'locator': head.state_dict(), 'locator_threshold': cutoff,
                    'intent_threshold': .5, 'seed': SEED, 'input_hashes': before}, OUT / 'candidate.pt')
        after = {k: base.sha(v) for k, v in paths.items()}
        result = {'rows': results, 'trace': trace, 'locator_cutoff': cutoff, 'intent_cutoff': .5,
                  'input_hashes': before, 'input_hashes_after': after, 'candidate_sha256': base.sha(OUT / 'candidate.pt'),
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Authored small sample; previous set informed design; no natural-data evaluation.',
                                  'Manual identity enrollment; no memory read, generation, or persistence.',
                                  'Joint retraining changes locator and intent; component effects not isolated.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected input changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds'], 'cutoff': cutoff}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
