"""Measure incremental state writes bound to a real structural checkpoint."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from spora.integrated_checkpoint import save_integrated
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerReplacement

BOOTSTRAP = ROOT/'runs/deepseek-spora-structural-checkpoint-20261111/individual-with-replacements.pt'
OUTPUT = ROOT/'runs/deepseek-spora-integrated-durable-probe-v2-20261111'
DATABASE = OUTPUT/'individual.sqlite'
FULL = OUTPUT/'full-comparison.pt'
ROWS = ({'question': '小苏现在喜欢什么饮品？', 'key': 9, 'intent': 'current', 'event_time': 3, 'expected': '可乐'},
        {'question': '小苏以前喜欢什么饮品？', 'key': 9, 'intent': 'past', 'event_time': 2, 'expected': '牛奶'})


def base_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    return base, tokenizer


def measure(runtime, tokenizer):
    results = []
    with torch.inference_mode():
        for row in ROWS:
            prefix = closed_think_inputs(tokenizer, row['question'], 'cuda')
            args = _memory_args(runtime.organism, runtime.state, row)
            args.pop('memory_state')
            generated = runtime.generate(**prefix, **args, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = generated[0, prefix['input_ids'].shape[1]:].tolist()
            answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
            results.append({'question': row['question'], 'intent': row['intent'], 'expected': row['expected'],
                            'answer': answer, 'token_ids': tail,
                            'exact_correct': answer in (row['expected'], row['expected']+'.', row['expected']+'。')})
    return {'rows': results, 'state_digest': frozen_digest(runtime.organism, runtime.state),
            'cursor': runtime.state.cursor, 'revision': runtime.store.revision}


def verify():
    torch.set_num_threads(2)
    base, tokenizer = base_and_tokenizer()
    with load_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda') as runtime:
        result = measure(runtime, tokenizer)
        result['replacement_lengths'] = {str(i): runtime.decoder.base.model.layers[i].max_length for i in (6, 12)}
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight mismatch')
    source_report = json.loads((BOOTSTRAP.parent/'result.json').read_text(encoding='utf-8'))
    bootstrap_sha = sha(BOOTSTRAP)
    if source_report['checkpoint_sha256'] != bootstrap_sha:
        raise ValueError('Structural bootstrap digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base, tokenizer = base_and_tokenizer()
    OUTPUT.mkdir(parents=False, exist_ok=False)
    init_started = time.perf_counter()
    runtime = create_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda')
    init_seconds = time.perf_counter()-init_started
    writes = []
    for value in (2, 5):
        torch.cuda.synchronize()
        before = time.perf_counter()
        decision, stats = runtime.observe_sourced(9, value, 'direct', 1.0)
        torch.cuda.synchronize()
        if not decision['current_write']:
            raise ValueError('Controlled memory write rejected')
        writes.append({'value': value, 'action': decision['action'], 'seconds': time.perf_counter()-before, **stats})
        print(json.dumps({'write': len(writes), 'tensor_bytes_written': stats['tensor_bytes_written'],
                          'seconds': writes[-1]['seconds']}), flush=True)
    before_reload = measure(runtime, tokenizer)
    wal = DATABASE.with_name(DATABASE.name+'-wal')
    database_bytes_before_close = DATABASE.stat().st_size
    wal_bytes_before_close = wal.stat().st_size if wal.exists() else 0
    torch.cuda.synchronize()
    full_started = time.perf_counter()
    replacements = {i: layer for i, layer in enumerate(runtime.decoder.base.model.layers)
                    if isinstance(layer, SPORALayerReplacement)}
    save_integrated(FULL, runtime.organism, runtime.state, runtime.decoder.reader, MODEL,
                    runtime.decoder.layer_indices, replacements=replacements)
    torch.cuda.synchronize()
    full_seconds = time.perf_counter()-full_started
    full_bytes = FULL.stat().st_size
    runtime.close()
    del runtime, base, tokenizer, replacements
    gc.collect()
    torch.cuda.empty_cache()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify'],
                           capture_output=True, text=True, encoding='utf-8', timeout=360, check=True)
    after = json.loads(child.stdout.strip().splitlines()[-1])
    if before_reload['state_digest'] != after['state_digest'] or before_reload['cursor'] != after['cursor']:
        raise ValueError('Durable state changed after independent reload')
    same = [a['token_ids'] == b['token_ids'] for a, b in zip(before_reload['rows'], after['rows'])]
    if sha(BOOTSTRAP) != bootstrap_sha or sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {'writes': writes, 'before': before_reload, 'after_independent_process': after,
              'same_tokens': same, 'bootstrap_sha256': bootstrap_sha, 'base_sha256': WEIGHT_SHA,
              'database_bytes_before_close': database_bytes_before_close, 'wal_bytes_before_close': wal_bytes_before_close,
              'database_bytes_after_close': DATABASE.stat().st_size, 'full_checkpoint_bytes': full_bytes,
              'full_checkpoint_seconds': full_seconds, 'initial_load_and_store_seconds': init_seconds,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two synthetic writes and two caller-addressed short questions.',
                              'Tensor bytes are logical payload, not total SQLite or device I/O.',
                              'Single process measurements with uncontrolled external GPU/disk load.',
                              'Bootstrap hashes verified on open; in-session stat and tensor-version checks are not tamper proof.',
                              'Separate frozen base and structural bootstrap files remain necessary.',
                              'Independent reload does not by itself simulate power loss or prove open-chat ability.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'same_tokens': same, 'correct_before': sum(r['exact_correct'] for r in before_reload['rows']),
                      'correct_after': sum(r['exact_correct'] for r in after['rows'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=True), flush=True)
    else:
        run()
