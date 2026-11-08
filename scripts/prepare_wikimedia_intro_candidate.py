"""Extract traceable lead paragraphs for review, never auto-approve training."""
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import sha

SOURCE = ROOT/'runs/research/wikimedia-wikipedia-zh/shard-00002.parquet'
SOURCE_SHA = 'ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f'
VOCAB = ROOT/'runs/p28-stories-data/candidate-v3/tokenizer.json'
DEV = ROOT/'data/conversation-development.json'
OUT = ROOT/'runs/wikimedia-intro-candidate-v3'
NOISE = re.compile(r'https?://|www\.|\{\{|\}\}|\[\[|\]\]|-\{[^}]+\}-|\(\s*[,;]?\s*\)|'
                   r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|'
                   r'可以指[:：]|可能指[:：]|是指以下|消歧义|消歧義|^列表|\b(?:isbn|doi)\b|'
                   r'(?:治疗|治療|药物|藥物|草药|草藥|防病|杀菌|殺菌|疗效|療效|患者|病症|'
                   r'患病|罹患|自杀|自殺|自残|自殘|疾病|癌症|抑郁|憂鬱|细胞|細胞|'
                   r'函数|函數|数学|數學|公式|向量|矩阵|矩陣|映射|同态|同態|维度|維度|'
                   r'轨域|軌域|群论|群論|代数|代數)|'
                   r'\(\s*[,;、:：\- ]+\s*\)', re.I)


def norm(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def extract(text):
    text = unicodedata.normalize('NFKC', text or '').strip()
    # Stop before later headings, lists, references and categories.
    intro = re.split(r'\n\s*\n', text, maxsplit=1)[0].strip()
    intro = re.sub(r'\s+', ' ', intro)
    if not 120 <= len(intro) <= 1200 or NOISE.search(intro):
        return None
    if re.match(r'^(?:是|为|為|可能是|通常是)[\s,，、。]', intro) or intro.startswith(('是一个','是一個','是一部','是一种','是一種')):
        return None
    if sum('\u4e00' <= c <= '\u9fff' for c in intro) / len(intro) < .45:
        return None
    if sum(c.isdigit() for c in intro) / len(intro) > .10:
        return None
    return intro


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Source checksum mismatch')
    tokenizer = Tokenizer.from_file(str(VOCAB))
    cases = json.loads(DEV.read_text(encoding='utf8'))
    prompts = [norm(p) for case in cases['scenarios'] for p in case['turns'] if len(norm(p)) >= 12]
    seen, counts = set(), collections.Counter()
    OUT.mkdir(parents=True)
    streams = {s: ((OUT/f'{s}.jsonl').open('x', encoding='utf8'), (OUT/f'{s}.bin').open('xb'))
               for s in ('train', 'development')}
    try:
        for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=256, columns=['id','url','title','text']):
            for row in batch.to_pylist():
                counts['source_rows'] += 1
                if not row['id'] or not row['url']:
                    counts['missing_source'] += 1
                    continue
                if '消歧' in (row['title'] or ''):
                    counts['title_disambiguation'] += 1
                    continue
                text = extract(row['text'])
                if not text:
                    counts['mechanical_reject'] += 1
                    continue
                value = norm(text)
                if any(p in value for p in prompts):
                    counts['known_dev_full_containment'] += 1
                    continue
                group = hashlib.sha256(value.encode('utf8')).hexdigest()
                if group in seen:
                    counts['exact_duplicate'] += 1
                    continue
                seen.add(group)
                split = 'development' if int(group[:8],16) % 20 == 0 else 'train'
                ids = [tokenizer.token_to_id('<|bos|>')] + tokenizer.encode(text).ids + [tokenizer.token_to_id('<|eos|>')]
                if len(ids) < 3 or len(ids) > 8192:
                    counts['token_length'] += 1
                    continue
                item = dict(group=group, page_id=row['id'], url=row['url'], title=row['title'], text=text)
                streams[split][0].write(json.dumps(item, ensure_ascii=False)+'\n')
                np.asarray(ids, dtype=np.uint16).tofile(streams[split][1])
                counts[split+'_rows'] += 1
                counts[split+'_tokens'] += len(ids)
    finally:
        for text_file, binary in streams.values():
            text_file.close()
            binary.close()
    if not counts['train_rows'] or not counts['development_rows']:
        raise ValueError('Empty candidate split')
    manifest = dict(status='candidate_not_approved_for_training', source_repository='wikimedia/wikipedia',
                    revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa', source_sha256=SOURCE_SHA,
                    declared_license='CC-BY-SA-3.0 and GFDL per dataset card; preserve page attribution',
                    tokenizer_sha256=sha(VOCAB), generator_sha256=sha(Path(__file__)),
                    development_cases_sha256=sha(DEV), counts=counts,
                    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
                    limitations=['A single sequential shard is not representative of all Wikipedia.',
                                 'Machine filters do not guarantee factual accuracy, coherence or absence of identifying information.',
                                 'Exact paragraph dedup and grouped split do not catch article-level or near-duplicate leakage.',
                                 'Only full containment of known development prompts checked; final tests not read.',
                                 'Source and output require content, isolation and license review before any training.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
