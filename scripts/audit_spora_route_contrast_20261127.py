"""Evaluate saved name locator on paired read, negation and unknown requests."""
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
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

PARENT = ROOT / 'runs/spora-isolated-name-routes-v1'
OUT = ROOT / 'runs/research/route-contrast-20261127-v1'


def cases():
    result = []
    for i, name in enumerate(NAMES[:8]):
        other = NAMES[i ^ 1]
        longer = name + '宁'
        if longer in NAMES or longer == other:
            raise ValueError('Unknown identity conflicts with enrollment')
        examples = [('read', f'还记得{name}平常爱喝什么饮料吗？', i),
                    ('excluded', f'不要查询{name}喜欢喝什么，请问二加三是多少？', None),
                    ('word_task', f'“{name}”这个名字有几个字？', None),
                    ('unknown_longer', f'还记得{longer}平常爱喝什么饮料吗？', None),
                    ('two_people', f'请分别告诉我{name}和{other}平常爱喝什么饮料。', None)]
        for group, text, expected in examples:
            result.append({'group': group, 'text': text, 'target': None, 'expected': expected})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'locator': PARENT / 'locator.pt', 'report': PARENT / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        report = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['locator'] != report['locator_sha256']:
            raise ValueError('Locator digest differs from original report')
        original_sources = report['sources']
        for key in ('base', 'adapter'):
            matches = [value for path, value in original_sources.items()
                       if Path(path).name == paths[key].name]
            if len(matches) != 1 or matches[0] != before[key]:
                raise ValueError('Source digest differs: ' + key)
        cutoff = report['cutoff']
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        adapter = torch.load(paths['adapter'], weights_only=True, map_location='cpu')
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        head = MentionHead(model.config.hidden_size).eval()
        head.load_state_dict(torch.load(paths['locator'], map_location='cpu', weights_only=True))
        bank = IdentityFingerprints(len(NAMES))
        for key, name in enumerate(NAMES):
            bank.bind(key, name)
        rows = cases()
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        print(json.dumps({'encoding': len(rows)}), flush=True)
        found = locate(head, encode(extractor, rows), rows)
        decisions = []
        for i, (row, hit) in enumerate(zip(rows, found, strict=True)):
            key = route(bank, row['text'], hit, cutoff)
            decisions.append({'index': i, 'group': row['group'], 'expected_key': row['expected'],
                              'selected_key': key, 'correct': key == row['expected'],
                              'margin': hit['margin']})
        after = {k: base.sha(v) for k, v in paths.items()}
        result = {'rows': decisions, 'cutoff': cutoff, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Small authored examples; existing enrolled names and previously developed candidate.',
                                  'Single-object routing only; rejection is not a general conversational response.',
                                  'No memory access, generation, persistence, or autonomous write.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if after != before:
            raise ValueError('Protected input changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
