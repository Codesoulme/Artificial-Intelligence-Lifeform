"""Create a quarantined, grouped Chinese dialogue candidate from WildChat."""
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
from scripts.sample_wildchat_chinese import RISK
from spora.dialogue_windows import dialogue_windows

SOURCE = ROOT / 'runs/research/wildchat/shard-00008.parquet'
SOURCE_SHA = 'a69578183dde5de1aa94134da2d7c2d13f00d88a577ef91140a492fbbbf10792'
VOCAB = ROOT / 'runs/p28-stories-data/candidate-v3/tokenizer.json'
DEV = ROOT / 'data/conversation-development.json'
OUT = ROOT / 'runs/wildchat-chinese-candidate-v3'
EXTRA_RISK = re.compile(r'我叫|我名叫|我的名字|本人名叫|我的姓名|本人姓名|姓名是|姓名为|姓名為|'
                        r'openai|chatgpt|gpt-?[345]|我是.{0,10}(?:语言模型|語言模型|人工智能助手)|'
                        r'精油|护肤|護膚|眼霜|抗衰|美白|祛斑', re.I)


def norm(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def shingles(text):
    return {text[i:i + 5] for i in range(len(text) - 4)}


def group_similar(rows):
    """Union exact or high-Jaccard first prompts; does not catch paraphrases."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    postings, exact, sets = collections.defaultdict(list), {}, []
    for i, row in enumerate(rows):
        key = row['prompt_norm']
        if key in exact:
            union(i, exact[key])
        else:
            exact[key] = i
        grams = shingles(key) if len(key) >= 8 else set()
        sets.append(grams)
        candidates = set()
        for gram in grams:
            candidates.update(postings[gram])
        for j in candidates:
            other = sets[j]
            union_size = len(grams | other)
            if union_size and len(grams & other) / union_size >= .80:
                union(i, j)
        for gram in grams:
            postings[gram].append(i)
    members = collections.defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    return list(members.values())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Source checksum mismatch')
    tokenizer = Tokenizer.from_file(str(VOCAB))
    known = json.loads(DEV.read_text(encoding='utf8'))
    dev_prompts = [norm(p) for scenario in known['scenarios'] for p in scenario['turns']]
    dev_prompts = [p for p in dev_prompts if len(p) >= 12]
    columns = ['language', 'toxic', 'redacted'] + [
        'conversation.list.element.' + f for f in ('content', 'role', 'toxic', 'redacted')]
    rows, seen, counts = [], set(), collections.Counter()
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=128, columns=columns):
        for source in batch.to_pylist():
            if source['language'] != 'Chinese':
                continue
            counts['chinese_labeled'] += 1
            if source['toxic'] is not False or source['redacted'] is not False:
                counts['source_flag'] += 1
                continue
            turns = source['conversation'] or []
            if not 4 <= len(turns) <= 8 or len(turns) % 2:
                counts['turn_count'] += 1
                continue
            if any(t.get('role') != ('user' if i % 2 == 0 else 'assistant') or
                   t.get('redacted') is not False or t.get('toxic') is not False
                   for i, t in enumerate(turns)):
                counts['turn_flags'] += 1
                continue
            messages = [{'role': t['role'], 'content': (t.get('content') or '').strip()} for t in turns]
            text = ''.join(t['content'] for t in messages)
            if any(not t['content'] or len(t['content']) > 1800 or RISK.search(t['content']) or
                   EXTRA_RISK.search(t['content']) for t in messages):
                counts['content_filter'] += 1
                continue
            if sum('\u4e00' <= c <= '\u9fff' for c in text) / len(text) < .40:
                counts['language_ratio'] += 1
                continue
            prompts = [norm(m['content']) for m in messages if m['role'] == 'user']
            if any(p in q or (len(q) >= 12 and q in p) for p in dev_prompts for q in prompts):
                counts['known_dev_containment'] += 1
                continue
            encoded = encode_dialogue(tokenizer, messages, limit=8192)
            if encoded is None:
                counts['too_long'] += 1
                continue
            identity = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
            if identity in seen:
                counts['duplicate'] += 1
                continue
            seen.add(identity)
            tokens, mask = encoded
            rows.append(dict(identity=identity, prompt_norm=prompts[0], messages=messages, tokens=tokens, mask=mask))
    groups = group_similar(rows)
    splits = {'train': [], 'development': []}
    for indices in groups:
        # Content-derived group identity is independent of input ordering.
        group = hashlib.sha256('\n'.join(sorted(rows[i]['identity'] for i in indices)).encode('ascii')).hexdigest()
        split = 'development' if int(group[:8], 16) % 10 == 0 else 'train'
        for i in indices:
            row = rows[i]
            windows = list(dialogue_windows(row['tokens'], row['mask'], length=512, overlap=128))
            expected = sum(row['mask'][1:])
            actual = sum(sum(mask[1:]) for _, _, mask in windows)
            if actual != expected:
                raise AssertionError('Windowed target coverage differs')
            splits[split].append(({'group': group, 'messages': row['messages'], 'identity': row['identity']},
                                  [dict(group=group, identity=row['identity'], offset=offset,
                                        tokens=tokens, mask=mask) for offset, tokens, mask in windows]))
    if not all(splits.values()):
        raise ValueError('Empty candidate split')
    OUT.mkdir(parents=True)
    for split, items in splits.items():
        with (OUT / (split + '.jsonl')).open('x', encoding='utf8') as f:
            for raw, _ in items:
                f.write(json.dumps(raw, ensure_ascii=False) + '\n')
        with (OUT / (split + '-tokens.jsonl')).open('x', encoding='utf8') as f:
            for _, windows in items:
                for window in windows:
                    f.write(json.dumps(window, ensure_ascii=False) + '\n')
    manifest = dict(status='candidate_not_approved_for_training', source_repository='allenai/WildChat-1M',
                    revision='7d6490e462285cf85d91eabea0f9a954fbddcd1f', source_sha256=SOURCE_SHA,
                    declared_license='ODC-BY', tokenizer_sha256=sha(VOCAB), generator_sha256=sha(Path(__file__)),
                    screening_source_sha256=sha(ROOT/'scripts/sample_wildchat_chinese.py'),
                    window_source_sha256=sha(ROOT/'spora/dialogue_windows.py'),
                    development_cases_sha256=sha(DEV), filter_counts=counts, groups=len(groups),
                    split_counts={k: len(v) for k, v in splits.items()},
                    window_counts={k: sum(len(w) for _, w in v) for k, v in splits.items()},
                    supervised_targets={k: sum(sum(sum(window['mask'][1:]) for window in w) for _, w in v)
                                        for k, v in splits.items()},
                    files={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
                    limitations=['Heuristic filters do not guarantee privacy, factual accuracy, safety, or identity consistency.',
                                 'Long conversations use overlapping 512-token windows; later windows lack earlier context beyond 128 overlap tokens.',
                                 'Assistant targets are supervised once per original dialogue, but training samples windows rather than whole dialogues.',
                                 'First-prompt 5-gram Jaccard grouping misses paraphrases and later-turn duplicates.',
                                 'Only exact full containment with known development prompts checked; final tests not read.',
                                 'Assistant text was externally generated; no external model is called during SPORA inference.',
                                 'Not approved for training; human quality and independent leakage reviews required.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
