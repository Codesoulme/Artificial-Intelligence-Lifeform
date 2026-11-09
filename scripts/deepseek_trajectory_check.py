"""Check a DeepSeek memory adapter on held-out multi-event SPORA trajectories."""
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_full_answer import thinking_inputs
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import native_read
from scripts.open_base_trajectory_comparison import build_trajectories, numeric_correct
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


@torch.inference_mode()
def generate(model, tokenizer, question, layers=None, adapter=None, state=None, addresses=None, reasoning=False):
    inputs = thinking_inputs(tokenizer, question, 'cuda') if reasoning else closed_think_inputs(tokenizer, question, 'cuda')
    limit = 64 if reasoning else 12
    if adapter is None:
        output = model.generate(**inputs, max_new_tokens=limit, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses):
            output = model.generate(**inputs, max_new_tokens=limit, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    answer = (raw.split('</think>', 1)[1].strip() if '</think>' in raw else None) if reasoning else raw
    return {'raw': raw, 'answer': answer}


def run(adapter_path, output, reasoning=False):
    adapter_path, output = Path(adapter_path).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not adapter_path.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Adapter and new output must be under runs')
    adapter_sha = sha(adapter_path)
    checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
    expected_format = ('deepseek-spora-reasoning-full-answer-v1' if reasoning else
                       'deepseek-spora-closed-think-full-answer-v1')
    if (checkpoint.get('format') != expected_format
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('query_sha256') != QUERY_SHA or checkpoint.get('layers') != [5, 17]
            or checkpoint.get('current_only') is not True):
        raise ValueError('Adapter provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(QUERY_PATH) == QUERY_SHA and sha(adapter_path) == adapter_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(4605)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    reader, reader_state, _ = load(QUERY_PATH, 'cpu')
    if any(not torch.equal(a.current_basis, b.current_basis) for a, b in zip(organism.memories, reader.memories)):
        raise ValueError('Incompatible neural addresses')
    before, reader_before = frozen_digest(organism, original), frozen_digest(reader, reader_state)
    trajectories = build_trajectories(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    for item in trajectories:
        for key in item['keys']:
            question = f'对象{key}的值是多少？'
            target = item['targets'][key]
            native = native_read(organism, item['state'], [key])[0]
            row = {'trajectory': item['id'], 'key': key, 'question': question, 'expected': target,
                   'native': native, 'native_correct': native == target,
                   'baseline': generate(model, tokenizer, question, reasoning=reasoning)}
            try:
                addresses, parsed = neural_addresses(reader, organism, question)
                if parsed['entity'] != key:
                    raise ValueError('Wrong parsed entity')
            except ValueError as exc:
                row.update(address_accepted=False, error=str(exc), combined_correct=False)
            else:
                response = generate(model, tokenizer, question, layers, adapter, item['state'], addresses, reasoning)
                row.update(address_accepted=True, combined=response,
                           combined_correct=response['answer'] is not None and numeric_correct(response['answer'], target))
            rows.append(row)
        print(f"trajectory {item['id'] + 1}/4 complete", flush=True)
    if not verify() or frozen_digest(organism, original) != before or frozen_digest(reader, reader_state) != reader_before:
        raise ValueError('Protected source changed')
    summary = {'examples': len(rows), 'native_correct': sum(r['native_correct'] for r in rows),
               'combined_correct': sum(r['combined_correct'] for r in rows),
               'address_accepted': sum(r['address_accepted'] for r in rows),
               'baseline_correct': sum(r['baseline']['answer'] is not None and
                                       numeric_correct(r['baseline']['answer'], r['expected']) for r in rows)}
    report = {'status': 'controlled_trajectory_diagnostic_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'query_sha256': QUERY_SHA,
              'adapter_sha256': adapter_sha, 'summary': summary, 'rows': rows,
              'trajectories': [{k: v for k, v in item.items() if k != 'state'} for item in trajectories],
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All trajectories share one original background and address basis.',
                              ('Fixed short reasoning texts were supervised; no multi-step reasoning is tested.' if reasoning else
                               'The think section is explicitly closed; no reasoning is tested.'),
                              'All numerical values were in adapter training.',
                              'P26 may have seen held keys and prompts.',
                              'Exact numeric matching may reject correct full-sentence responses.',
                              'Baseline never receives event data; its score is an ablation, not a fair memory comparison.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reasoning', action='store_true')
    args = parser.parse_args()
    run(args.adapter, args.output, args.reasoning)
