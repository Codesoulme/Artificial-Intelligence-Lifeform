"""Isolated cross-process relation persistence audit with actual DeepSeek weights."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_structured_runtime_pilot import BOOTSTRAP, load_base, make_codec
from spora.integrated_durable import create_integrated_durable, load_integrated_durable

OUTPUT = ROOT / 'runs/deepseek-spora-relation-durable-audit-20261113'
DATABASE = OUTPUT / 'individual.sqlite'
META = (('drink', '茶'), ('hobby', '读书'), ('location', '北京'))


def encoded(codec, tokenizer, fact):
    identity = tokenizer.encode('小林', add_special_tokens=False)
    value = tokenizer.encode(fact, add_special_tokens=False)
    packet, metadata = codec.encode(identity, value)
    return packet, metadata, identity, value


def inspect(runtime, codec, tokenizer):
    result = {}
    for relation, expected in META:
        record = runtime.read_structured_relation(4, relation)
        decoded = codec.decode(record['packet'], record['metadata'])
        _, _, identity, fact = encoded(codec, tokenizer, expected)
        result[relation] = {'identity_exact': decoded['identity']['token_ids'] == identity,
                            'fact_exact': decoded['fact']['token_ids'] == fact,
                            'revision': record['revision']}
    legacy = runtime.read_structured_text(0)
    decoded = codec.decode(legacy['packet'], legacy['metadata'])
    _, _, identity, fact = encoded(codec, tokenizer, '苹果')
    return {'relations': result,
            'legacy_exact': decoded['identity']['token_ids'] == identity and decoded['fact']['token_ids'] == fact,
            'unknown_absent': runtime.read_structured_relation(4, 'unknown') is None,
            'other_entity_absent': runtime.read_structured_relation(5, 'drink') is None,
            'store_revision': runtime.store.revision,
            'bindings': len(runtime.state.structured_text.relations)}


def run_verify():
    torch.set_num_threads(2)
    base, tokenizer = load_base()
    with load_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda') as runtime:
        codec = make_codec(runtime, tokenizer)
        return inspect(runtime, codec, tokenizer)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight mismatch')
    source_hash = sha(BOOTSTRAP)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    OUTPUT.mkdir(parents=False, exist_ok=False)
    base, tokenizer = load_base()
    with create_integrated_durable(DATABASE, BOOTSTRAP, base, MODEL, individual_device='cuda') as runtime:
        codec = make_codec(runtime, tokenizer)
        packet, metadata, _, _ = encoded(codec, tokenizer, '苹果')
        runtime.write_structured_text(0, packet, metadata, codec_config=codec.config_dict())
        writes = []
        for relation, fact in [('drink', '咖啡'), ('hobby', '读书'), ('location', '北京')]:
            packet, metadata, _, _ = encoded(codec, tokenizer, fact)
            writes.append(runtime.write_structured_relation(4, relation, packet, metadata))
        packet, metadata, _, _ = encoded(codec, tokenizer, '茶')
        writes.append(runtime.write_structured_relation(4, 'drink', packet, metadata, replace_existing=True))
        before = inspect(runtime, codec, tokenizer)
    del runtime, codec, tokenizer, base
    gc.collect()
    torch.cuda.empty_cache()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify'],
                           capture_output=True, timeout=360, check=False)
    if child.returncode:
        raise RuntimeError(child.stderr.decode('utf-8', errors='replace')[-4000:])
    after = json.loads(child.stdout.decode('utf-8', errors='replace').strip().splitlines()[-1])
    if before != after or not all(row['identity_exact'] and row['fact_exact'] for row in after['relations'].values()) or not all((after['legacy_exact'], after['unknown_absent'], after['other_entity_absent'])):
        raise ValueError('Cross-process relation check failed')
    if sha(BOOTSTRAP) != source_hash or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {'status': 'isolated_deepseek_relation_persistence_audit', 'before': before,
              'after': after, 'writes': writes, 'bootstrap_sha256': source_hash,
              'base_sha256': WEIGHT_SHA, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Explicit entity and relation; no identity or semantic routing.',
                              'Exact codebook decoding is not generated language.',
                              'No historical record, long-term stress or real user state.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(run_verify(), ensure_ascii=False), flush=True)
    else:
        run()
