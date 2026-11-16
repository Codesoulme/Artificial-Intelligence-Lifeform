"""Compare normal-reasoning generation across two SPORA position tails.

This development audit does not activate persistent memory or store generated
reasoning text. Final answers, token counts, termination and timing are logged.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_position_diverse_distill as dense
from scripts import deepseek_spora_smooth_position_distill as smooth
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from spora.structure_transfer import apply_position_tails

OUT = ROOT / 'runs/deepseek-spora-position-free-generation-20261116.json'
MAX_NEW = 448
SEED = 58131
LIMIT = 512


def validate_tail(payload, sources, widths):
    keys = set(widths)
    if (not isinstance(payload, dict) or payload.get('format') != 'spora-position-tail-v1'
            or payload.get('base_weight_sha256') != base.WEIGHT_SHA or payload.get('sources') != sources
            or payload.get('old_length') != 128 or payload.get('new_length') != LIMIT
            or not isinstance(payload.get('tails'), dict) or set(payload['tails']) != keys):
        raise ValueError('Position-tail binding mismatch')
    for key, width in widths.items():
        value = payload['tails'][key]
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.shape != (LIMIT - 128, width) or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid position-tail tensor')


def summarize_tail(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    closed = '</think>' in text
    return {'reasoning_closed': closed, 'final_answer': text.split('</think>', 1)[1].strip() if closed else ''}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    dense_report = json.loads((dense.OUT / 'result.json').read_text(encoding='utf-8'))
    smooth_report = json.loads((smooth.OUT / 'result.json').read_text(encoding='utf-8'))
    candidates = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    sources = {str(index): base.sha(path) for index, path in candidates.items()}
    tail_paths = {'dense': dense.OUT / 'position-tails.pt', 'smooth': smooth.OUT / 'position-tails.pt'}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or dense_report['base_sha256'] != base.WEIGHT_SHA or smooth_report['base_sha256'] != base.WEIGHT_SHA
            or dense_report['source_sha256'] != sources or smooth_report['source_sha256'] != sources
            or base.sha(tail_paths['dense']) != dense_report['position_tails_sha256']
            or base.sha(tail_paths['smooth']) != smooth_report['position_tails_sha256']):
        raise ValueError('Base, replacement or tail provenance mismatch')
    payloads = {name: torch.load(path, map_location='cpu', weights_only=True)
                for name, path in tail_paths.items()}
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    _, held = data.load_reviewed_prompts()
    cases = [held[1], held[7]]
    if len({case['group'] for case in cases}) != len(cases):
        raise ValueError('Dialogue sources overlap')
    inputs = []
    for case in cases:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}], tokenize=False,
                                                add_generation_prompt=True)
        if not prompt.rstrip().endswith('<think>'):
            raise ValueError('Unexpected normal-reasoning template')
        encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        if encoded['input_ids'].shape[1] + MAX_NEW > LIMIT:
            raise ValueError('Prompt plus generation exceeds replacement limit')
        inputs.append(encoded)
    original_layers = {index: model.model.layers[index] for index in candidates}
    blocks = {index: _load_candidate(path, index, model).eval() for index, path in candidates.items()}
    widths = {str(index): block.width for index, block in blocks.items()}
    for payload in payloads.values():
        validate_tail(payload, sources, widths)
    apply_position_tails(blocks, payloads['dense'], base.WEIGHT_SHA, sources)
    prefix_hash = {str(index): hashlib.sha256(block.position.weight[:128].detach().cpu().numpy().tobytes()).hexdigest()
                   for index, block in blocks.items()}
    rows = []
    for mode in ('base', 'zero', 'dense', 'smooth'):
        for index in candidates:
            model.model.layers[index] = original_layers[index] if mode == 'base' else blocks[index]
        if mode != 'base':
            with torch.no_grad():
                for index, block in blocks.items():
                    if mode == 'zero':
                        block.position.weight[128:].zero_()
                    else:
                        block.position.weight[128:].copy_(payloads[mode]['tails'][str(index)].to(
                            device=block.position.weight.device, dtype=block.position.weight.dtype))
        for index, (case, encoded) in enumerate(zip(cases, inputs)):
            torch.manual_seed(SEED + index)
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**encoded, do_sample=True, temperature=.6, top_p=.95,
                                        max_new_tokens=MAX_NEW, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tail = output[0, encoded['input_ids'].shape[1]:].detach().cpu()
            summary = summarize_tail(tokenizer, tail.tolist())
            row = {'mode': mode, 'source_group': case['group'], 'source_line': case['line'],
                   'prompt_sha256': hashlib.sha256(case['prompt'].encode('utf-8')).hexdigest(),
                   'prompt_tokens': encoded['input_ids'].shape[1], 'seed': SEED + index,
                   'generated_tokens': len(tail), 'ended_with_eos': bool(len(tail) and tail[-1] == tokenizer.eos_token_id),
                   'generated_token_sha256': hashlib.sha256(tail.numpy().tobytes()).hexdigest(),
                   'seconds': time.perf_counter() - before, **summary}
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('mode', 'source_line', 'generated_tokens',
                                                       'reasoning_closed', 'ended_with_eos', 'seconds')}), flush=True)
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(path) != sources[str(index)] for index, path in candidates.items())
            or base.sha(tail_paths['dense']) != dense_report['position_tails_sha256']
            or base.sha(tail_paths['smooth']) != smooth_report['position_tails_sha256']):
        raise ValueError('Protected input changed')
    if any(hashlib.sha256(block.position.weight[:128].detach().cpu().numpy().tobytes()).hexdigest()
           != prefix_hash[str(index)] for index, block in blocks.items()):
        raise ValueError('Source positions changed')
    report = {'base_sha256': base.WEIGHT_SHA, 'candidate_sha256': sources,
              'tail_sha256': {name: base.sha(path) for name, path in tail_paths.items()}, 'rows': rows,
              'settings': {'max_new_tokens': MAX_NEW, 'temperature': .6, 'top_p': .95,
                           'position_limit': LIMIT, 'system_prompt': False},
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two previously reviewed questions, one seed per question and configuration.',
                              'Normal sampling may diverge even with shared seeds; not a significance test.',
                              '448-token cap can truncate reasoning or final answers.',
                              'No active SPORA memory, autonomous learning or multi-turn evaluation.',
                              'Positions beyond 351 are extrapolated or zero and were not trained directly.',
                              'Only final answers and output hashes retained; reasoning text omitted.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'results': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
