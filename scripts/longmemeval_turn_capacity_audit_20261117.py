"""Aggregate token lengths; no gold labels, training or content output."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from transformers import AutoTokenizer
from scripts.deepseek_local_baseline import MODEL
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn

INPUT = ROOT / 'runs/research/longmemeval-oracle-prepared-20261117/input.jsonl'
OUT = ROOT / 'runs/longmemeval-turn-capacity-audit-20261117.json'


def summarize_lengths(lengths, *, capacity=4, max_chunks=64):
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('Lengths must be positive integers')
    if type(capacity) is not int or capacity < 1 or type(max_chunks) is not int or max_chunks < 1:
        raise ValueError('Invalid capacity')
    return {'count': len(lengths), 'above_one_packet': sum(n > capacity for n in lengths),
            'median_tokens': statistics.median(lengths), 'max_tokens': max(lengths),
            'median_packets': statistics.median((n + capacity - 1) // capacity for n in lengths),
            'over_address_capacity': sum(n > capacity * max_chunks for n in lengths),
            'tokens_per_packet': capacity, 'address_count': max_chunks}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    raw = INPUT.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    turns = [first_user_turn(row) for row in rows]
    if any(text is None for text in turns):
        raise ValueError('A case has no user turn')
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in turns]
    result = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'summary': summarize_lengths(lengths),
              'limitations': ['First nonempty user turn per oracle case only.',
                              'English benchmark conversations, not representative of Chinese use.',
                              'Counts are storage estimates, not measured retrieval performance.',
                              'One packet per address assumes no sharing and ignores other records.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
