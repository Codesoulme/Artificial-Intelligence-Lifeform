"""Build a family-disjoint review sample from long-dialogue candidates."""
import hashlib
import json
from pathlib import Path

from scripts import prepare_p231_dialogue_candidates as previous
from tokenizers import Tokenizer

ROOT = previous.ROOT
INPUT = ROOT/'runs/p235-long-dialogue-candidates-20261109'
OUTPUT = ROOT/'runs/p235-long-dialogue-review-20261109'
SEED = 'p235-review-20261109-v1'


def rank(kind, value):
    return hashlib.sha256((SEED+'\0'+kind+'\0'+value).encode()).hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((INPUT/'manifest.json').read_bytes())
    raw = (INPUT/'candidates.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['candidate_sha256']:
        raise ValueError('Candidate changed')
    old_manifest = json.loads((ROOT/'runs/p231-dialogue-candidates/manifest.json').read_bytes())
    prior = ROOT/'runs/p231-dialogue-candidates/review-40.jsonl'
    if previous.file_hash(prior) != old_manifest['output_sha256']['review-40.jsonl']:
        raise ValueError('Prior sample changed')
    excluded = {json.loads(line)['family_sha256'] for line in prior.read_bytes().splitlines()}
    families = {}
    for raw_line in raw.splitlines():
        row = json.loads(raw_line)
        family = row['family_sha256']
        if family not in excluded:
            families.setdefault(family, []).append(row)
    chosen = []
    for family in sorted(families, key=lambda f: rank('family', f))[:20]:
        chosen.append(min(families[family], key=lambda r: rank('turn', r['dialogue_sha256']+':'+str(r['answer_message_index']))))
    if len(chosen) != 20:
        raise ValueError('Insufficient disjoint families')
    source = ROOT/'runs/conversation-data-v5/dialogue-train.jsonl'
    tokenizer_path = ROOT/'runs/conversation-data-v5/tokenizer.json'
    if previous.file_hash(source) != previous.EXPECTED['source'] or previous.file_hash(tokenizer_path) != previous.EXPECTED['tokenizer']:
        raise ValueError('Source or tokenizer changed')
    wanted = {r['source_line'] for r in chosen}
    origins = {}
    hasher = hashlib.sha256()
    with source.open('rb') as stream:
        for n, line in enumerate(stream, 1):
            hasher.update(line)
            if n in wanted:
                origins[n] = json.loads(line)
    if hasher.hexdigest() != previous.EXPECTED['source'] or len(origins) != len(wanted):
        raise ValueError('Source changed or missing selected row')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sample = []
    for number, row in enumerate(chosen):
        origin = origins[row['source_line']]
        if previous.dialogue_hash(origin['messages']) != row['dialogue_sha256']:
            raise ValueError('Source dialogue differs')
        answer = row['answer_message_index']
        sample.append({'index': number, 'family_sha256': row['family_sha256'], 'dialogue_sha256': row['dialogue_sha256'],
                       'answer_message_index': answer, 'context_start_message_index': row['context_start_message_index'],
                       'messages': origin['messages'][:answer+1],
                       'decoded_target': tokenizer.decode([t for t, m in zip(row['tokens'], row['mask']) if m], skip_special_tokens=False)})
    payload = b''.join((json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8') for row in sample)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    (OUTPUT/'sample.jsonl').write_bytes(payload)
    meta = {'status': 'review_only_not_training', 'seed': SEED, 'samples': len(sample),
            'sampling': 'hash-ranked families excluding P231 sample families; one answer per family',
            'candidate_sha256': manifest['candidate_sha256'], 'prior_sample_sha256': previous.file_hash(prior),
            'source_sha256': previous.EXPECTED['source'], 'tokenizer_sha256': previous.EXPECTED['tokenizer'],
            'script_sha256': previous.file_hash(Path(__file__)), 'sample_sha256': hashlib.sha256(payload).hexdigest(),
            'limitations': ['Family selection is not uniform over answer turns.', 'Prior reviews beyond P231 are not excluded.',
                            'The sample contains full prior dialogue and must remain local.']}
    (OUTPUT/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return meta


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
