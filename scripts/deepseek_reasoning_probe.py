"""Probe a closed-think-trained adapter with DeepSeek's original reasoning template."""
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

from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import variants, pack_states
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


def run(adapter_path, output):
    adapter_path, output = Path(adapter_path).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not adapter_path.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Adapter and new output must be under runs')
    adapter_sha = sha(adapter_path)
    checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-closed-think-full-answer-v1'
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
    torch.manual_seed(4606)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    reader, reader_state, _ = load(QUERY_PATH, 'cpu')
    if any(not torch.equal(a.current_basis, b.current_basis) for a, b in zip(organism.memories, reader.memories)):
        raise ValueError('Incompatible neural addresses')
    before, reader_before = frozen_digest(organism, original), frozen_digest(reader, reader_state)
    states, _ = variants(organism, original, ((8, 9),))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    question = '对象8的值是多少？'
    addresses, parsed = neural_addresses(reader, organism, question)
    if parsed['entity'] != 8:
        raise ValueError('Wrong neural address')
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                               add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
    inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
    rows = []
    with torch.inference_mode():
        for mode in ('disabled', 'memory'):
            t0 = time.perf_counter()
            if mode == 'disabled':
                output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
            else:
                with memory_hooks(layers, adapter, pack_states(states[(8, 9)]), current_only=True, addresses=addresses):
                    output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            for index, target in enumerate((1, 2)):
                tokens = output_ids[index, inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                rows.append({'mode': mode, 'state_index': index, 'expected': target,
                             'generated_tokens_in_batch': int(tokens.numel()),
                             'ended_with_eos': tokenizer.eos_token_id in tokens.tolist(),
                             'think_closed': '</think>' in text, 'generated_text': text,
                             'final_answer': text.split('</think>', 1)[1].strip() if '</think>' in text else None,
                             'batch_seconds': elapsed})
            print(f'{mode} complete', flush=True)
    if not verify() or frozen_digest(organism, original) != before or frozen_digest(reader, reader_state) != reader_before:
        raise ValueError('Protected source changed')
    report = {'status': 'reasoning_template_development_probe_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'query_sha256': QUERY_SHA,
              'adapter_sha256': adapter_sha, 'question': question, 'rows': rows,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two states share one original background and one numbered question.',
                              'Adapter was trained with the thinking section explicitly closed.',
                              'Greedy decoding and 384-token maximum are diagnostic, not recommended DeepSeek settings.',
                              'A final answer is not automatically judged correct.',
                              'No evidence of general reasoning, unknown refusal or new experiences.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{k: v for k, v in row.items() if k != 'generated_text'} for row in rows], ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.adapter, args.output)
