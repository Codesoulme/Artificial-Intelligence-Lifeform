"""Build a local, unapproved review sample from the pinned P75 training side."""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs' / 'wildchat-chinese-candidate-v3'
MANIFEST_SHA = '3f0c22e1b02e42d3c38b3beaa18f22c4be88e9023ec9b84cecc4873e7c28cc89'
TRAIN_SHA = '835a60eeb3e993cb84949e376d1efa4546e78ef0fc58962b3acdb4b9ce8bc3b9'
OUT = ROOT / 'runs' / 'wildchat-private-review-v1'

# Quarantine whole groups. These are deliberately broad and are not anonymization.
PATTERNS = {
    'contact_or_link': r'https?\s*[:：]|www\.|[\w.+-]+\s*@\s*[\w.-]+|微信|微訊|电话|電話|手机|手機|邮箱|郵箱|联系我|聯繫我|QQ号|账号|賬號',
    'number_or_identifier': r'(?<!\d)\d[\d\s+().-]{5,}\d(?!\d)|身份证|身分證|护照|護照|银行卡|銀行卡|信用卡|车牌|車牌|密码|密碼|验证码|驗證碼|住址|家庭地址|详细地址|郵編|邮编',
    'personal_name': r'我叫|我名叫|我的名字|本人名叫|我的姓名|本人姓名|姓名[是为為]|真名|实名|實名|联系人|聯絡人|收件人',
    'sensitive_context': r'病史|患者|诊断|診斷|病历|病歷|心理咨询|心理諮詢|自杀|自殺|自残|自殘|性侵|怀孕|懷孕|艾滋|工资|工資|收入|债务|債務|银行账户|銀行賬戶|个人信息|個人信息',
    'personal_narrative': r'我(?:的|在|住|就读|就讀|工作于|工作於|出生|毕业于|畢業於|今年|岁|歲|患有|被诊断|被診斷|家住|来自|來自)|本人|我的(?:家人|孩子|父亲|母亲|丈夫|妻子)',
}
FILTERS = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode('utf8')).hexdigest()


def norm(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def validate_groups(rows):
    """Rebuild P75 connected components and check complete member-derived groups."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parents[max(a, b)] = min(a, b)
    exact, postings, sets, identities = {}, collections.defaultdict(list), [], set()
    for i, row in enumerate(rows):
        messages = row['messages']
        if len(messages) not in (4, 6, 8) or any(
            set(m) != {'role', 'content'} or m['role'] != ('user' if j % 2 == 0 else 'assistant')
            or not isinstance(m['content'], str) or not m['content'] for j, m in enumerate(messages)
        ):
            raise ValueError('Invalid dialogue structure')
        identity = digest(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        if identity != row['identity'] or identity in identities:
            raise ValueError('Invalid or duplicate dialogue identity')
        identities.add(identity)
        key = norm(messages[0]['content'])
        if key in exact:
            union(i, exact[key])
        else:
            exact[key] = i
        grams = {key[j:j+5] for j in range(len(key)-4)} if len(key) >= 8 else set()
        sets.append(grams)
        candidates = set()
        for gram in grams:
            candidates.update(postings[gram])
        for j in candidates:
            if len(grams & sets[j]) / len(grams | sets[j]) >= .8:
                union(i, j)
        for gram in grams:
            postings[gram].append(i)
    components = collections.defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = {}
    for members in components.values():
        group = digest('\n'.join(sorted(r['identity'] for r in members)))
        if any(r['group'] != group for r in members) or int(group[:8], 16) % 10 == 0:
            raise ValueError('Group membership or training split cannot be verified')
        groups[group] = members
    return groups


def risks(row):
    text = unicodedata.normalize('NFKC', '\n'.join(m['content'] for m in row['messages']))
    return {name for name, pattern in FILTERS.items() if pattern.search(text)}


def rank(seed, value):
    return digest(f'{seed}\n{value}')


def select(groups, previous, count, seed):
    buckets, excluded = collections.defaultdict(list), collections.Counter()
    for group, members in sorted(groups.items()):
        if group in previous:
            excluded['prior_review_group'] += 1
            continue
        labels = set().union(*(risks(r) for r in members))
        if labels:
            excluded.update(labels)
            excluded['privacy_any_group'] += 1
            continue
        # A fixed representative per group avoids overweighting large groups.
        row = min(members, key=lambda r: rank(seed, r['identity']))
        chars = sum(len(m['content']) for m in row['messages'])
        bucket = f"{len(row['messages'])//2}_turns_" + ('short' if chars < 1000 else 'long')
        buckets[bucket].append((group, row))
    for bucket, items in buckets.items():
        items.sort(key=lambda pair: rank(seed, bucket + pair[0]))
    sizes = {k: len(v) for k, v in sorted(buckets.items())}
    if sum(sizes.values()) < count:
        raise ValueError('Insufficient eligible groups')
    selected = []
    while len(selected) < count:
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < count:
                group, row = buckets[bucket].pop(0)
                selected.append((bucket, group, row))
    return selected, sizes, dict(sorted(excluded.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--seed', type=int, default=861109)
    args = parser.parse_args()
    if not 1 <= args.count <= 200:
        raise ValueError('Count must be between 1 and 200')
    # No caller-provided paths: never read another split or write outside runs.
    if OUT.exists() or OUT.is_symlink() or (ROOT / 'runs').is_symlink():
        raise ValueError('Output exists or runs is a symlink')
    if subprocess.run(['git', 'check-ignore', '-q', '--', 'runs/wildchat-private-review-v1/sample.jsonl'], cwd=ROOT).returncode:
        raise ValueError('Output is not ignored by Git')
    manifest_path, train_path, prior_path = BASE/'manifest.json', BASE/'train.jsonl', BASE/'review-sample.json'
    if sha(manifest_path) != MANIFEST_SHA or sha(train_path) != TRAIN_SHA:
        raise ValueError('Pinned manifest or training file checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['status'] != 'candidate_not_approved_for_training' or manifest['files']['train.jsonl'] != TRAIN_SHA:
        raise ValueError('Unexpected source status or training checksum')
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    if prior['manifest_sha256'] != MANIFEST_SHA or prior['source_sha256'] != TRAIN_SHA:
        raise ValueError('Previous review provenance mismatch')
    rows = [json.loads(line) for line in train_path.read_text(encoding='utf8').splitlines()]
    if len(rows) != manifest['split_counts']['train']:
        raise ValueError('Training row count mismatch')
    groups = validate_groups(rows)
    previous = {s['group'] for s in prior['samples']}
    if len(previous) != prior['sample_size'] or not previous <= groups.keys():
        raise ValueError('Previous review groups cannot be verified')
    selected, strata, excluded = select(groups, previous, args.count, args.seed)
    OUT.mkdir()
    sample_path = OUT/'sample.jsonl'
    with sample_path.open('x', encoding='utf8', newline='\n') as f:
        for bucket, group, row in selected:
            f.write(json.dumps({'stratum': bucket, 'group': group, 'identity': row['identity'],
                                'messages': row['messages']}, ensure_ascii=False) + '\n')
    result = {
        'status': 'review_sample_not_approved_for_training', 'source_repository': manifest['source_repository'],
        'revision': manifest['revision'], 'declared_license': manifest['declared_license'],
        'source_shard_sha256': manifest['source_sha256'], 'source_manifest_sha256': MANIFEST_SHA,
        'training_file_sha256': TRAIN_SHA, 'prior_review_file_sha256': sha(prior_path),
        'generator_sha256': sha(Path(__file__)), 'sample_sha256': sha(sample_path),
        'seed': args.seed, 'training_records': len(rows), 'training_groups': len(groups),
        'excluded_groups': excluded, 'eligible_groups_by_stratum': strata,
        'selected_groups_by_stratum': dict(sorted(collections.Counter(x[0] for x in selected).items())),
        'sample_size': len(selected), 'privacy_patterns': PATTERNS,
        'limitations': ['Heuristics can miss identifiers and combinations of indirect clues; no anonymity guarantee.',
                        'Group-level exclusion can remove benign content; categories overlap.',
                        'First-prompt grouping misses paraphrases, later-turn duplicates and cross-source overlap.',
                        'Strata are balanced and groups sampled once; proportions are not population estimates.',
                        'No independent test content accessed; no claim about contamination or training approval.',
                        'Review full conversations locally for privacy, quality, safety, provenance and permission.']}
    (OUT/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ('status', 'training_records', 'training_groups', 'excluded_groups',
          'eligible_groups_by_stratum', 'selected_groups_by_stratum', 'sample_size')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
