"""Compare a bounded BELLE file prefix with an existing dialogue training pool."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def fingerprint(text):
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())
    return hashlib.sha256(text.encode('utf-8')).digest()


def audit(prefix_path, training_path):
    prefix = prefix_path.read_bytes()
    lines = prefix.splitlines()
    # The last line may be partial when an HTTP byte range ends mid-record.
    complete_lines = lines if prefix.endswith((b'\n', b'\r')) else lines[:-1]
    rows = [json.loads(line) for line in complete_lines]
    prior_first, prior_answers = set(), set()
    hasher = hashlib.sha256()
    with training_path.open('rb') as source:
        for line in source:
            hasher.update(line)
            messages = json.loads(line)['messages']
            prior_first.add(fingerprint(messages[0]['content']))
            prior_answers.update(fingerprint(m['content']) for m in messages if m['role'] == 'assistant')
    firsts, answers = [], []
    for row in rows:
        instruction, answer = row['instruction'], row['output']
        if not instruction.startswith('Human:') or not instruction.rstrip().endswith('Assistant:'):
            raise ValueError('Unexpected dialogue boundary')
        parts = re.split(r'(?:^|\n)(Human|Assistant):', instruction)
        if len(parts) < 3 or parts[1] != 'Human' or not parts[2].strip():
            raise ValueError('Cannot parse first user message')
        firsts.append(fingerprint(parts[2]))
        answers.append(fingerprint(answer))
    return {'prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'training_file_sha256': hasher.hexdigest(),
            'complete_rows': len(rows), 'unique_first_prompts': len(set(firsts)),
            'unique_outputs': len(set(answers)),
            'rows_first_prompt_matching_train': sum(x in prior_first for x in firsts),
            'rows_output_matching_train': sum(x in prior_answers for x in answers),
            'method': 'NFKC, casefold and remove whitespace; SHA-256 exact comparison',
            'limitations': 'Sequential prefix, not random; first prompt and output matches do not establish complete dialogue duplication; no fuzzy, cross-source or final-test comparison.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prefix, args.training)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
