"""Prepare a non-approved, prompt-grouped Chinese Aya dialogue candidate."""
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import encode_dialogue, sha

SOURCE = ROOT / 'runs/research/aya/train.parquet'
SOURCE_SHA = '51baa85043b569ba117f41516b32d1b0e4e2647fb203a08933f98a556fb1fb06'
VOCAB = ROOT / 'runs/p28-stories-data/candidate-v3/tokenizer.json'
DEVELOPMENT = ROOT / 'data/conversation-development.json'
OUT = ROOT / 'runs/aya-chinese-candidate-v1'
RISK = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|https?://|<unk>|疾病|药物|药品|处方|诊断|治疗|手术|剂量|驾车|驾驶|法律|律师|自杀|自傷|自残', re.I)


def normalized(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text or '').casefold())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Source checksum mismatch')
    tokenizer = Tokenizer.from_file(str(VOCAB))
    development = json.loads(DEVELOPMENT.read_text(encoding='utf8'))
    protected = [normalized(p) for case in development['scenarios'] for p in case['turns']]
    protected = [p for p in protected if len(p) >= 12]
    counts, samples, seen = collections.Counter(), {'train': [], 'development': []}, set()
    columns = ['inputs', 'targets', 'language', 'language_code', 'annotation_type']
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            if row['language_code'] != 'zho' or row['language'] != 'Simplified Chinese' or row['annotation_type'] != 'original-annotations':
                continue
            counts['simplified_original_rows'] += 1
            prompt, answer = (row['inputs'] or '').strip(), (row['targets'] or '').strip()
            norm_prompt, norm_answer = normalized(prompt), normalized(answer)
            if not (6 <= len(prompt) <= 500 and 8 <= len(answer) <= 700 and norm_prompt and norm_answer):
                counts['length_or_empty'] += 1
                continue
            if RISK.search(prompt + '\n' + answer):
                counts['risk_pattern'] += 1
                continue
            if len(norm_answer) >= 8 and norm_answer in norm_prompt:
                counts['answer_in_prompt'] += 1
                continue
            if any(p in norm_prompt or (len(norm_prompt) >= 12 and norm_prompt in p) for p in protected):
                counts['development_full_containment'] += 1
                continue
            key = (norm_prompt, norm_answer)
            if key in seen:
                counts['duplicate_pair'] += 1
                continue
            seen.add(key)
            group = hashlib.sha256(norm_prompt.encode('utf8')).hexdigest()
            split = 'development' if int(group[:8], 16) % 10 == 0 else 'train'
            messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
            encoded = encode_dialogue(tokenizer, messages, limit=512)
            if encoded is None:
                counts['token_length'] += 1
                continue
            tokens, mask = encoded
            samples[split].append(({'group': group, 'messages': messages},
                                   {'group': group, 'tokens': tokens, 'mask': mask}))
    groups = [{row[0]['group'] for row in samples[split]} for split in ('train', 'development')]
    if not all(samples.values()) or groups[0] & groups[1]:
        raise ValueError('Empty candidate or overlapping prompt groups')
    OUT.mkdir(parents=True)
    for split, rows in samples.items():
        for suffix, column in (('.jsonl', 0), ('-tokens.jsonl', 1)):
            with (OUT / (split + suffix)).open('x', encoding='utf8') as f:
                for row in rows:
                    f.write(json.dumps(row[column], ensure_ascii=False) + '\n')
    manifest = dict(status='candidate_not_approved_for_training', source_repository='CohereForAI/aya_dataset',
                    revision='f9ea04583f02a8f86404ff6c58bf75fe637df8a2', source_sha256=SOURCE_SHA,
                    declared_license='Apache-2.0', tokenizer_sha256=sha(VOCAB), generator_sha256=sha(Path(__file__)),
                    development_cases_sha256=sha(DEVELOPMENT), filter_counts=counts,
                    split_counts={s: len(rows) for s, rows in samples.items()},
                    files={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
                    limitations=['Single-turn original annotations only; automatic screening does not verify facts or safety.',
                                 'Prompt grouping blocks exact normalized duplicates across splits, not paraphrases or shared annotator patterns.',
                                 'Excludes only exact normalized containment with known development prompts; final tests not read.',
                                 'Requires full provenance, quality and leakage review before any training.'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
