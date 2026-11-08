"""Bounded, metadata-only screening of the pinned MiniMind training partition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/conversation-data-v5/dialogue-train.jsonl'
EXPECTED = '2629f0c8668ef6736c2a0431f0681c24944299e5d60c8ae8ac78d688b565973e'
SEED = 'p104-20261109-v1'
PATTERNS = {
    'contact': re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)'),
    'long_number': re.compile(r'(?<!\d)\d{12,}(?!\d)'),
    'identity': re.compile(r'我叫[\u4e00-\u9fff]{2,4}|身份证|家庭住址|银行卡|密码'),
    'sensitive': re.compile(r'自杀|自残|强奸|未成年.{0,8}性|制造.{0,5}(?:炸弹|毒品)'),
    'experience': re.compile(r'我(?:昨天|今天|昨晚|上周|去年|小时候|亲自|曾经|在工作中|在学校|的家人)'),
    'control': re.compile(r'<\||```|[\x00-\x08\x0b\x0c\x0e-\x1f]'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def assess(row):
    messages = row.get('messages')
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        return ['shape']
    for i, msg in enumerate(messages):
        if (not isinstance(msg, dict) or msg.get('role') != ('user' if i % 2 == 0 else 'assistant')
                or not isinstance(msg.get('content'), str) or not msg['content'].strip()):
            return ['shape']
    text = unicodedata.normalize('NFKC', '\n'.join(m['content'] for m in messages))
    reasons = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    if len(text) > 12000:
        reasons.append('length')
    if sum('\u4e00' <= c <= '\u9fff' for c in text) < 40:
        reasons.append('few_chinese')
    return reasons


def screen(rows, limit=500, seed=SEED):
    """Group by original key OR identical normalized first prompt, before filtering."""
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    groups, prompts = {}, {}
    for i, row in enumerate(rows):
        group = row.get('group')
        if not isinstance(group, str) or not group:
            raise ValueError('missing original group')
        if group in groups:
            union(i, groups[group])
        groups[group] = i
        msgs = row.get('messages')
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get('content'), str):
            prompt = norm(msgs[0]['content'])
            if prompt:
                if prompt in prompts:
                    union(i, prompts[prompt])
                prompts[prompt] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    ranked = []
    for members in components.values():
        if not any(isinstance(rows[i].get('messages'), list) and len(rows[i]['messages']) >= 4 for i in members):
            continue
        key = digest('\n'.join(sorted(set(rows[i]['group'] for i in members))).encode())
        ranked.append((digest((seed + '\0' + key).encode()), key, members))
    ranked.sort()
    counts = Counter()
    selected = []
    # Bound applies to groups inspected, including discarded groups.
    for _, key, members in ranked[:limit]:
        counts['groups_examined'] += 1
        flags = set()
        for i in members:
            flags.update(assess(rows[i]))
        if flags:
            counts['groups_quarantined'] += 1
            counts.update('flag_' + flag for flag in flags)
            continue
        split = 'development' if int(digest((seed + '\0split\0' + key).encode())[:8], 16) % 5 == 0 else 'candidate'
        counts[split + '_groups'] += 1
        for i in members:
            if len(rows[i]['messages']) < 4:
                counts['single_turn_rows_excluded'] += 1
                continue
            counts[split + '_rows'] += 1
            counts[split + '_assistant_characters'] += sum(len(m['content']) for m in rows[i]['messages'] if m['role'] == 'assistant')
            selected.append({'line': i + 1, 'group_sha256': key, 'partition': split})
    return dict(counts), selected, len(components)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, default=500, choices=range(1, 2001), metavar='1..2000')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/P104-screen.json')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent != (ROOT / 'reports').resolve() or not out.name.startswith('P104') or out.suffix != '.json':
        parser.error('output must be reports/P104*.json')
    if out.exists():
        raise FileExistsError(out)
    raw = SOURCE.read_bytes()
    actual = digest(raw)
    if actual != EXPECTED:
        raise ValueError('training file digest mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    counts, selected, total = screen(rows, args.groups)
    result = {'source': 'runs/conversation-data-v5/dialogue-train.jsonl', 'source_sha256': actual,
              'script_sha256': digest(Path(__file__).read_bytes()), 'seed': SEED, 'group_limit': args.groups,
              'source_rows': len(rows), 'source_groups_after_exact_prompt_union': total,
              'counts': counts, 'rows': selected,
              'limitations': ['Metadata-only index; not a training file.', 'Filters are heuristic and do not prove privacy, accuracy or rights.',
                              'Only exact first prompts and original group keys are joined; semantic overlap is unknown.',
                              'Characters are not model tokens; source may overlap prior training and development.']}
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
