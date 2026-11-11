"""Test bounded native-thinking on raw and 768-position integrated layers."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_pair_oasst_reviewed_eval import DATA_SHA, REVIEW_SHA, selected_prompts
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-native-think-768-audit-20261112.json'
LINES = (19, 26, 28, 70)
MAX_NEW = 256
PARITY_NEW = 24


def generate(base, tokenizer, inputs, budget, cache):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        ids = base.generate(**inputs, max_new_tokens=budget, do_sample=False, use_cache=cache,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    return ids[0, inputs['input_ids'].shape[1]:].tolist(), time.perf_counter() - started


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(runtime.CHECKPOINT)
    items = {r['line']: r for r in selected_prompts()}
    if any(line not in items for line in LINES):
        raise ValueError('Reviewed prompt unavailable')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    close = tokenizer('</think>', add_special_tokens=False)['input_ids']
    if len(close) != 1:
        raise ValueError('Unexpected thinking delimiter')
    encoded = []
    for line in LINES:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': items[line]['question']}],
                                             tokenize=False, add_generation_prompt=True)
        if '<think>' not in text or '</think>' in text:
            raise ValueError('Unexpected native thinking prompt')
        encoded.append(tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda'))
    base = load_full_streamed().eval()
    outputs = {}
    parity = []
    lengths = {}
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        for mode in ('raw', 'integrated_768'):
            if mode == 'integrated_768':
                chat = runtime.load_chat(base_loader=lambda: base)
                original_digest = frozen_digest(chat.organism, chat.state)
                replacements = {i: layer for i, layer in enumerate(base.model.layers)
                                if isinstance(layer, SPORALayerReplacement)}
                if set(replacements) != {6, 12}:
                    raise ValueError('Unexpected integrated replacements')
                lengths = {str(i): {'positions': layer.max_length,
                                    'nonzero_tail_rows': int((layer.position.weight[128:].detach().abs().sum(-1) > 0).sum())}
                           for i, layer in replacements.items()}
                if any(x['positions'] < MAX_NEW + max(e['input_ids'].shape[1] for e in encoded)
                       for x in lengths.values()):
                    raise ValueError('Insufficient replacement position capacity')
                for line, inputs in zip(LINES[:2], encoded[:2]):
                    cached, cache_seconds = generate(base, tokenizer, inputs, PARITY_NEW, True)
                    uncached, no_cache_seconds = generate(base, tokenizer, inputs, PARITY_NEW, False)
                    parity.append({'line': line, 'tokens_match': cached == uncached,
                                   'cached_tokens': len(cached), 'uncached_tokens': len(uncached),
                                   'cached_seconds': cache_seconds, 'uncached_seconds': no_cache_seconds})
                if not all(r['tokens_match'] for r in parity):
                    raise ValueError('Short cache parity failed')
            rows = []
            for line, inputs in zip(LINES, encoded):
                tail, seconds = generate(base, tokenizer, inputs, MAX_NEW, True)
                end = bool(tail and tokenizer.eos_token_id is not None and tail[-1] == tokenizer.eos_token_id)
                idx = tail.index(close[0]) if close[0] in tail else None
                final_ids = [] if idx is None else tail[idx + 1:]
                if final_ids and end:
                    final_ids = final_ids[:-1]
                final = tokenizer.decode(final_ids, skip_special_tokens=True).strip() if final_ids else ''
                row = {'line': line, 'question': items[line]['question'], 'input_tokens': inputs['input_ids'].shape[1],
                       'generated_tokens': len(tail), 'thinking_closed': idx is not None,
                       'tokens_before_close': idx, 'has_final_answer': bool(final),
                       'final_answer': final, 'ended_with_eos': end, 'seconds': seconds}
                rows.append(row)
                print(json.dumps({'mode': mode, 'line': line, 'closed': row['thinking_closed'],
                                  'has_final': row['has_final_answer'], 'eos': end, 'seconds': round(seconds, 2)}), flush=True)
            outputs[mode] = rows
        if frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Protected individual changed in memory')
    if sha(runtime.CHECKPOINT) != source_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source file changed')
    summary = {mode: {'cases': len(rows), 'thinking_closed': sum(r['thinking_closed'] for r in rows),
                      'has_final_answer': sum(r['has_final_answer'] for r in rows),
                      'ended_with_eos': sum(r['ended_with_eos'] for r in rows),
                      'generation_seconds': sum(r['seconds'] for r in rows)} for mode, rows in outputs.items()}
    report = {'status': 'native_think_integrated_768_development', 'summary': summary, 'rows': outputs,
              'short_cache_parity': parity, 'replacement_lengths': lengths, 'source_checkpoint_sha256': source_sha,
              'base_sha256': WEIGHT_SHA, 'data_sha256': DATA_SHA, 'review_sha256': REVIEW_SHA,
              'max_new_tokens': MAX_NEW, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously inspected public prompts; not an independent benchmark.',
                              'Reasoning text and generated token IDs are not persisted; final text may be partial.',
                              'Cache equality checked only on two prompts for 24 new tokens, not long cache parity.',
                              'Tail row nonzero counts do not establish training quality or usable context.',
                              'No memory reading/writing or background thinking evaluated.',
                              'Research streamed loader; raw and integrated runs use cache and greedy decoding.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'parity': parity, 'lengths': lengths}), flush=True)


if __name__ == '__main__':
    main()
