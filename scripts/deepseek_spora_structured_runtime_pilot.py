"""Cross-process structured-memory records through the real DeepSeek runtime."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.structured_text_memory import StructuredTextMemoryCodec


BOOTSTRAP = ROOT / 'runs/deepseek-spora-structural-checkpoint-20261111/individual-with-replacements.pt'
OUTPUT = ROOT / 'runs/continuous-reader-structured-runtime-pilot-v4-20261113'
DATABASE = OUTPUT / 'individual.sqlite'
NAMES = ('小林', '小周', '小王', '小赵')
WORDS = ('苹果', '香蕉', '西瓜', '葡萄')
CORRECTED_FACT = '草莓'


def load_base():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    return base, tokenizer


def make_codec(runtime, tokenizer):
    embedding = runtime.decoder.base.model.embed_tokens.weight.detach()
    return StructuredTextMemoryCodec(
        embedding, memory_width=runtime.organism.config.memory_width,
        identity_slots=3, fact_slots=5, identity_slot_width=15,
        fact_slot_width=9, seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda').eval()


def read_rows(runtime, codec, tokenizer):
    rows = []
    for key, name, word in zip(range(4), NAMES, WORDS):
        record = runtime.read_structured_text(key)
        if record is None:
            raise ValueError(f'Missing structured record {key}')
        decoded = codec.decode(record['packet'], record['metadata'])
        identity = tokenizer.decode(decoded['identity']['token_ids'], skip_special_tokens=True)
        fact = tokenizer.decode(decoded['fact']['token_ids'], skip_special_tokens=True)
        expected_identity_ids = tokenizer.encode(name, add_special_tokens=False)
        expected_fact_ids = tokenizer.encode(word, add_special_tokens=False)
        rows.append({'key': key, 'expected_identity': name, 'expected_fact': word,
                     'expected_identity_ids': expected_identity_ids,
                     'expected_fact_ids': expected_fact_ids,
                     'identity': identity, 'fact': fact,
                     'identity_ids': decoded['identity']['token_ids'],
                     'fact_ids': decoded['fact']['token_ids'],
                     'exact': (decoded['identity']['token_ids'] == expected_identity_ids
                               and decoded['fact']['token_ids'] == expected_fact_ids),
                     'revision': record['revision']})
    return rows


def verify():
    torch.set_num_threads(2)
    base, tokenizer = load_base()
    with load_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda') as runtime:
        codec = make_codec(runtime, tokenizer)
        rows = read_rows(runtime, codec, tokenizer)
        correction = runtime.read_structured_text(0)
        decoded = codec.decode(correction['packet'], correction['metadata'])
        corrected_fact = tokenizer.decode(decoded['fact']['token_ids'], skip_special_tokens=True)
        result = {'rows': rows, 'corrected_fact': corrected_fact,
                  'corrected_fact_ids': decoded['fact']['token_ids'],
                  'revision': runtime.store.revision, 'state_revision': runtime.state.structured_text.revision}
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight mismatch')
    bootstrap_sha = sha(BOOTSTRAP)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    OUTPUT.mkdir(parents=False, exist_ok=False)
    base, tokenizer = load_base()
    runtime = create_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda')
    codec = make_codec(runtime, tokenizer)
    config = codec.config_dict()
    writes = []
    for key, name, word in zip(range(4), NAMES, WORDS):
        identity_ids = tokenizer.encode(name, add_special_tokens=False)
        fact_ids = tokenizer.encode(word, add_special_tokens=False)
        packet, metadata = codec.encode(identity_ids, fact_ids)
        result = runtime.write_structured_text(key, packet, metadata, codec_config=config)
        writes.append({'key': key, 'name': name, 'word': word,
                       'identity_tokens': identity_ids, 'fact_tokens': fact_ids,
                       'state_revision': result['state_revision'],
                       'tensor_bytes_written': result['stats']['tensor_bytes_written']})
    old_packet, old_meta = codec.encode(
        tokenizer.encode(NAMES[0], add_special_tokens=False),
        tokenizer.encode(CORRECTED_FACT, add_special_tokens=False))
    runtime.write_structured_text(0, old_packet, old_meta, codec_config=config)
    before = read_rows(runtime, codec, tokenizer)
    before_revision = runtime.store.revision
    expected_corrected = tokenizer.decode(
        tokenizer.encode(CORRECTED_FACT, add_special_tokens=False), skip_special_tokens=True)
    expected_corrected_ids = tokenizer.encode(CORRECTED_FACT, add_special_tokens=False)
    runtime.close()
    del runtime, codec, tokenizer, base
    gc.collect()
    torch.cuda.empty_cache()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify'],
                           capture_output=True, timeout=360, check=False)
    if child.returncode != 0:
        error = child.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f'Independent structured reload failed: {error[-4000:]}')
    child_output = child.stdout.decode('utf-8', errors='replace').strip()
    if not child_output:
        raise RuntimeError('Independent structured reload produced no result')
    after = json.loads(child_output.splitlines()[-1])
    if [row['exact'] for row in before] != [row['exact'] for row in after['rows']]:
        raise ValueError('Structured rows changed after independent reload')
    if after['corrected_fact_ids'] != expected_corrected_ids:
        raise ValueError(f'Correction did not survive independent reload: {after["corrected_fact_ids"]!r} != {expected_corrected_ids!r}')
    if sha(BOOTSTRAP) != bootstrap_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {
        'status': 'deepseek_structured_runtime_pilot',
        'writes': writes,
        'before_close': {'rows': before, 'revision': before_revision},
        'after_independent_process': after,
        'same_exact_flags': [row['exact'] for row in before] == [row['exact'] for row in after['rows']],
        'correction_survived': after['corrected_fact_ids'] == expected_corrected_ids,
        'bootstrap_sha256': bootstrap_sha,
        'base_sha256': WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Temporary isolated TensorStore and explicit tokenizer sequences.',
            'No natural-language identity routing or generated answer conditioning.',
            'The structured codec is connected to the durable open-base runtime but is not yet the default chat write path.',
        ],
    }
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'same_exact_flags': report['same_exact_flags'],
                      'correction_survived': report['correction_survived']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False), flush=True)
    else:
        run()
