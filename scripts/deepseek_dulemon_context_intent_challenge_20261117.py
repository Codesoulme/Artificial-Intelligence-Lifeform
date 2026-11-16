"""Read-only beverage-name intent audit, comparing one turn and previous turn.

The corpus has no direct label for this task. Labels here are a single author's
inspection of 25 selected beverage-related questions, not official ground truth.
No dialogue text is written to the output.
"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts.deepseek_dulemon_persona_audit import normalize_turn
from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_intent_fixed_challenge_20261117 import GATE
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

OUT = ROOT / 'runs/deepseek-dulemon-context-intent-challenge-20261117.json'
POSITIVE = {(230, 1)}
KEYWORDS = ('喝', '饮料', '饮品', '咖啡', '茶', '果汁')


def select_dialogues(path):
    rows = []
    with zipfile.ZipFile(path) as z, z.open('DuLeMon/both/test.json') as stream:
        for session, line in enumerate(stream):
            conversation = json.loads(line).get('conversation', [])
            for turn in range(1, len(conversation)):
                current = normalize_turn(conversation[turn])
                previous = normalize_turn(conversation[turn - 1])
                if not current or not previous:
                    continue
                role, text, _ = current
                if not any(x in text for x in ('？', '?')) or not any(x in text for x in KEYWORDS):
                    continue
                digest = hashlib.sha256((role+'\0'+text+'\0'+previous[0]+'\0'+previous[1]).encode('utf-8')).hexdigest()
                rows.append({'session': session, 'turn': turn, 'role': role, 'text': text,
                             'previous_role': previous[0], 'previous_text': previous[1],
                             'read': (session, turn) in POSITIVE, 'sample_sha256': digest})
    if len(rows) != 25 or sum(r['read'] for r in rows) != 1:
        raise ValueError('Unexpected selected challenge')
    return sorted(rows, key=lambda r: r['sample_sha256'])


def summary(rows, field):
    return {'total': len(rows), 'positives': sum(r['read'] for r in rows),
            'true_accepts': sum(r['read'] and r[field] for r in rows),
            'false_accepts': sum(not r['read'] and r[field] for r in rows)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': base.MODEL/'model.safetensors', 'layer6': PAIR/'layer-6.pt',
             'layer12': PAIR/'layer-12.pt', 'gate': GATE/'gate.pt', 'archive': ARCHIVE}
    before = {k: base.sha(p) for k, p in paths.items()}
    gate_data = torch.load(paths['gate'], map_location='cpu', weights_only=True)
    provenance = json.loads((GATE/'result.json').read_text(encoding='utf-8'))
    if (before['archive'] != ARCHIVE_SHA or before['base'] != base.WEIGHT_SHA
            or before['gate'] != provenance['gate_sha256'] or gate_data['feature'] != 'early_mean'
            or gate_data['source_sha256'] != {k: before[k] for k in ('base', 'layer6', 'layer12')}):
        raise ValueError('Input provenance mismatch')
    cases = select_dialogues(ARCHIVE)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for i in (6, 12):
        model.model.layers[i] = _load_candidate(paths[f'layer{i}'], i, model).eval()
    model.requires_grad_(False)
    gate = nn.Linear(model.config.hidden_size, 1).eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    rows = []
    with torch.inference_mode():
        for case in cases:
            # The probe was trained on the current turn alone. Concatenating
            # a role-tagged prior turn is an OOD diagnostic, not a trained model.
            texts = (case['text'], f"{case['previous_role']}：{case['previous_text']}\n{case['role']}：{case['text']}")
            scores = []
            for text in texts:
                encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
                output = model.model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
                features = F.normalize(output.hidden_states[1][0].float().mean(0).cpu(), dim=-1)
                scores.append(float(gate(features).sigmoid().item()))
            rows.append({'session': case['session'], 'turn': case['turn'], 'role': case['role'],
                         'sample_sha256': case['sample_sha256'], 'read': case['read'],
                         'current_score': scores[0], 'context_score': scores[1],
                         'current_accept': scores[0] >= gate_data['threshold'],
                         'context_accept': scores[1] >= gate_data['threshold']})
    if any(base.sha(p) != before[k] for k, p in paths.items()):
        raise ValueError('Source changed')
    result = {'current_only': summary(rows, 'current_accept'), 'prior_turn': summary(rows, 'context_accept'),
              'threshold': gate_data['threshold'], 'rows': rows, 'source_sha256': before,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Task labels manually assigned by one model after reading cases; no independent review.',
                              'Only 25 selected questions, one positive; repeated sessions and previously used corpus.',
                              'Prior-turn input is out of distribution for this probe; no training or threshold tuning.',
                              'Finite beverage-name direct-answer path only, not general memory-retrieval eligibility.',
                              'Classifier only: no actual identity, time, memory reads, answer generation or writes.',
                              'Corpus distribution rights unverified; text remains local.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('current_only', 'prior_turn')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
