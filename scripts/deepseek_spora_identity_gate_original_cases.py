"""Read-only replay of the original default-runtime near-name challenge."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_default_near_name_audit import CASES
from scripts.deepseek_spora_identity_gate_probe import OUTPUT as GATE

OUTPUT = ROOT / 'runs/deepseek-spora-identity-gate-original-cases-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    gate_file = GATE / 'gate.pt'
    info = json.loads((GATE / 'result.json').read_text(encoding='utf-8'))
    if runtime.sha(gate_file) != info['checkpoint_sha256']:
        raise ValueError('Gate digest mismatch')
    data = torch.load(gate_file, map_location='cpu', weights_only=True)
    if runtime.sha(runtime.CHECKPOINT) != data['source_checkpoint_sha256']:
        raise ValueError('Feature source mismatch')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        features, masks = runtime.extract(chat.base, chat.tokenizer, [q for _, q in CASES], batch_size=2)
        gate = nn.Linear(data['state']['weight'].shape[1], 1).eval()
        gate.load_state_dict(data['state'], strict=True)
        with torch.no_grad():
            scores = gate(pool(features, masks).float()).squeeze(-1).sigmoid()
    rows = [{'group': group, 'question': question, 'score': float(score),
             'accepted': bool(score >= data['threshold'])}
            for (group, question), score in zip(CASES, scores)]
    summary = {'known_total': sum(r['group'] == 'known' for r in rows),
               'known_accepted': sum(r['group'] == 'known' and r['accepted'] for r in rows),
               'unknown_total': sum(r['group'] == 'unknown' for r in rows),
               'unknown_accepted': sum(r['group'] == 'unknown' and r['accepted'] for r in rows)}
    report = {'status': 'research_only_no_writes', 'summary': summary, 'rows': rows,
              'gate_sha256': runtime.sha(gate_file), 'source_checkpoint_sha256': data['source_checkpoint_sha256'],
              'limitations': ['Ten development queries reused from the original failure discovery.',
                              'Gate assessed alone; no live router integration or user state changed.',
                              'Calibration included related name suffixes; this is not independent evaluation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
