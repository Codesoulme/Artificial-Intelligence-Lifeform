"""Evaluate a frozen router using disjoint people and predeclared calibration."""
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
from scripts.audit_spora_isolated_name_routes import CAL_NAMES, TRAIN_NAMES, MentionHead, encode, locate, route
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
OUT = ROOT / 'runs/research/route-disjoint-calibration-20261127-v1'
FORMS = {
    'cal': [('read', '请回忆{a}以前喜欢喝什么。'),
            ('excluded', '不要查{a}的喜好，请说说天气。'),
            ('word_task', '请问“{a}”这个名字有几个汉字？'),
            ('unknown_prefix', '请回忆{u}以前喜欢喝什么。'),
            ('unknown_suffix', '请回忆{v}以前喜欢喝什么。'),
            ('two_people', '请分别回忆{a}和{b}喜欢什么饮品。')],
    'held': [('read', '你还记得{a}爱点哪种饮料吗？'),
             ('excluded', '暂时别回想{a}的口味，三加二等于多少？'),
             ('word_task', '把“{a}”逐字写出来。'),
             ('unknown_prefix', '你还记得{u}爱点哪种饮料吗？'),
             ('unknown_suffix', '你还记得{v}爱点哪种饮料吗？'),
             ('two_people', '{a}与{b}各自常喝什么？')],
}


def make_rows(names, forms):
    rows = []
    for i, name in enumerate(names):
        other = names[(i + 1) % len(names)]
        prefix, suffix = '宁' + name, name + '宁'
        if prefix in NAMES or suffix in NAMES or name == other:
            raise ValueError('Invalid enrolled or ambiguous unknown name')
        for group, form in forms:
            target = {'read': name, 'unknown_prefix': prefix, 'unknown_suffix': suffix}.get(group)
            rows.append({'text': form.format(a=name, b=other, u=prefix, v=suffix),
                         'group': group, 'target': target,
                         'expected_key': NAMES.index(name) if group == 'read' else None})
    return rows


def selected(row, cutoff):
    return row['raw_key'] if row['intent_pass'] and row['margin'] > cutoff else None


def summarize(rows, cutoff):
    result = {}
    for group in dict.fromkeys(row['group'] for row in rows):
        items = [r for r in rows if r['group'] == group]
        choices = [selected(r, cutoff) for r in items]
        result[group] = {'n': len(items),
            'correct_read': sum(r['expected_key'] is not None and c == r['expected_key'] for r, c in zip(items, choices)),
            'wrong_address': sum(c is not None and c != r['expected_key'] for r, c in zip(items, choices)),
            'false_reject': sum(r['expected_key'] is not None and c is None for r, c in zip(items, choices)),
            'false_activation': sum(r['expected_key'] is None and c is not None for r, c in zip(items, choices)),
            'intent_pass': sum(r['intent_pass'] for r in items),
            'span_exact': sum(r['span_exact'] for r in items)}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def stage(name):
        data = {'stage': name, 'seconds': time.perf_counter() - start}
        (OUT / 'stage.json').write_text(json.dumps(data), encoding='utf8')
        print(json.dumps(data), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'candidate': PARENT / 'candidate.pt', 'report': PARENT / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        previous = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['candidate'] != previous['candidate_sha256'] or any(before[k] != previous['input_hashes'][k] for k in ('base', 'adapter')):
            raise ValueError('Frozen input differs from original experiment')
        payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
        if payload['intent_threshold'] != .5:
            raise ValueError('Unexpected intent cutoff')
        cal_names, held_names = NAMES[8:16], NAMES[16:24]
        if set(cal_names) & set(held_names) or (set(cal_names) | set(held_names)) & (set(TRAIN_NAMES) | set(CAL_NAMES)):
            raise ValueError('Overlapping people')
        groups = {'cal': make_rows(cal_names, FORMS['cal']), 'held': make_rows(held_names, FORMS['held'])}
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
        for i, name in enumerate(NAMES):
            bank.bind(i, name)
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        outputs = {}
        with torch.no_grad():
            for label, examples in groups.items():
                stage('encoding_' + label)
                data = encode(extractor, examples)
                hits = locate(locator, data, examples)
                probabilities = torch.sigmoid(intent(data[0], data[1])).tolist()
                outputs[label] = [dict(group=row['group'], expected_key=row['expected_key'], text=row['text'],
                    raw_key=route(bank, row['text'], hit, -1000000.), margin=hit['margin'],
                    span=hit['name'], span_exact=hit['name'] == row['target'],
                    intent_probability=p, intent_pass=p >= .5)
                    for row, hit, p in zip(examples, hits, probabilities, strict=True)]
        stage('calibrating')
        candidates = sorted({-1000000., *(r['margin'] for r in outputs['cal'])})
        valid = []
        for cutoff in candidates:
            summary = summarize(outputs['cal'], cutoff)
            wrong = sum(v['wrong_address'] for v in summary.values())
            correct = sum(v['correct_read'] for v in summary.values())
            if wrong == 0:
                valid.append((correct, cutoff))
        cutoff = max(valid)[1] if valid else max(r['margin'] for r in outputs['cal'])
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after:
            raise ValueError('Frozen input changed')
        result = {'old_cutoff': payload['locator_threshold'], 'calibrated_cutoff': cutoff,
                  'calibration_candidates': len(candidates), 'valid_candidates': len(valid),
                  'summary': {label: {'old': summarize(rows, payload['locator_threshold']),
                                      'calibrated': summarize(rows, cutoff)} for label, rows in outputs.items()},
                  'rows': outputs, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
