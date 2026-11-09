"""Development stress test for a finite memory-intent gate and format guard."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.language_training import frozen_digest

CASES = (
    (1, 'current', '对象8当前的数值是多少？'),
    (1, 'current', '当前对象26存储的数字是几？'),
    (1, 'current', '请告诉我编号5现在的数值。'),
    (1, 'current', '我想知道对象63如今的值。'),
    (1, 'current', '查一下对象0记录的当前数值。'),
    (1, 'current', '编号32现在记录的数字是多少？'),
    (1, 'current', '对象19的最新数值是几？'),
    (1, 'current', '读出编号7的现值。'),
    (0, 'negated', '不要读取对象8的值。'),
    (0, 'negated', '我不想知道对象26的值。'),
    (0, 'history', '对象8曾经的值是多少？'),
    (0, 'history', '告诉我对象26的旧值。'),
    (0, 'history', '对象7最初记录的值是什么？'),
    (0, 'write', '删除对象8的记忆。'),
    (0, 'write', '将对象26的值清零。'),
    (0, 'write', '不要查询，改写对象7。'),
    (0, 'derived', '对象8的值是奇数吗？'),
    (0, 'derived', '对象26的数值大于零吗？'),
    (0, 'unsupported', '对象8的名字是什么？'),
    (0, 'unsupported', '对象26的记录来自哪里？'),
    (0, 'unsupported', '对象7是否存在？'),
    (0, 'unsupported', '请解释对象8是什么。'),
    (0, 'mixed', '先读取对象8再删除它。'),
    (0, 'mixed', '查询对象26并立即更新它。'),
    (0, 'malformed', '告诉我对象的最新值。'),
    (0, 'malformed', '对象的数字到底是多少？'),
)


def run(gate_path, output):
    gate_path, output = Path(gate_path).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not gate_path.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected gate and new output under runs')
    digest = sha(gate_path)
    checkpoint = torch.load(gate_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-current-intent-gate-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('threshold') != 0.5):
        raise ValueError('Gate provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(gate_path) == digest)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    gate = IntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(checkpoint['state'], strict=True)
    features, masks = extract(model, tokenizer, [row[2] for row in CASES])
    with torch.inference_mode():
        scores = gate(features.to('cuda'), masks.to('cuda')).sigmoid().cpu().tolist()
    rows = []
    for (label, category, question), score in zip(CASES, scores):
        form_ok = valid_finite_question_form(question)
        accepted = form_ok and score >= 0.5
        rows.append({'label': label, 'category': category, 'question': question, 'score': score,
                     'form_ok': form_ok, 'accepted': accepted, 'correct': accepted == bool(label)})
    if not verify() or frozen_digest(organism, original) != before:
        raise ValueError('Protected source changed')
    report = {'status': 'handwritten_adversarial_development_only', 'gate_sha256': digest, 'rows': rows,
              'examples': len(rows), 'correct': sum(row['correct'] for row in rows),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small hand-authored stress set, not independent broad deployment validation.',
                              'No memory is read or answer generated.',
                              'Finite format guard is symbolic and rejects unsupported forms by design.',
                              'No test of whether an in-range object has a recorded memory.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'examples': len(rows), 'correct': report['correct'],
                      'failures': [row for row in rows if not row['correct']]}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.gate, args.output)
