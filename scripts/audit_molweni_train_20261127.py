"""Aggregate-only audit of the official Molweni training split."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/research/molweni-source-20261127-v1'
SOURCE = DATA / 'MRC(withDiscourse)/train.json'
OUT = DATA / 'train-audit.json'
EXPECTED = 'dc9c374d85e1174188e13ea5dafc15ab7551ad7f234203996a5eb4683edbb8ac'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Unexpected source digest')
    data = json.loads(raw)['data']
    if data['title'] != 'train':
        raise ValueError('Unexpected split')
    dialogues = data['dialogues']
    counts, hist, qids = collections.Counter(), collections.Counter(), collections.Counter()
    duplicate_contexts = collections.Counter()
    for dialogue in dialogues:
        context, edus, qas = dialogue['context'], dialogue['edus'], dialogue['qas']
        counts['dialogues'] += 1
        counts['utterances'] += len(edus)
        counts['discourse_links'] += len(dialogue['relations'])
        hist[str(len({x['speaker'] for x in edus}))] += 1
        duplicate_contexts[hashlib.sha256(context.encode('utf8')).hexdigest()] += 1
        categories = set()
        for qa in qas:
            counts['questions'] += 1
            qids[qa['id']] += 1
            no_answer = qa['is_impossible']
            if not isinstance(no_answer, bool):
                raise ValueError('Unexpected label')
            counts['unanswerable' if no_answer else 'answerable'] += 1
            categories.add(no_answer)
            answers = qa['answers']
            if bool(answers) == no_answer:
                counts['answer_label_mismatch'] += 1
            for field in ('answers', 'plausible_answers'):
                for a in qa.get(field, []):
                    counts[field + '_spans'] += 1
                    start, text = a['answer_start'], a['text']
                    if not isinstance(start, int) or start < 0 or context[start:start + len(text)] != text:
                        counts[field + '_offset_mismatch'] += 1
            if qa.get('plausible_answers'):
                counts['questions_with_plausible_answers'] += 1
        if categories == {True, False}:
            counts['dialogues_both_labels'] += 1
    counts['duplicate_question_ids'] = sum(n - 1 for n in qids.values() if n > 1)
    counts['duplicate_context_rows'] = sum(n - 1 for n in duplicate_contexts.values() if n > 1)
    result = {'source_sha256': digest, 'counts': dict(counts),
              'speakers_per_dialogue': dict(sorted(hist.items(), key=lambda x: int(x[0]))),
              'limitations': ['Offsets checked literally; no semantic correctness or leakage validation.',
                              'Only train split accessed; English Ubuntu chat is not Chinese personal memory.',
                              'No source texts, question texts, names, or candidate answers exported.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
