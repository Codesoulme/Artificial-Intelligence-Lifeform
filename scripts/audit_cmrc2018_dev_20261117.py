"""Download pinned CMRC development split, audit offsets and question coverage locally."""
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = 'c0eb1b6ba219847457e6af3180da722bbeb656af'
BLOB_SHA1 = '94c1f2df14a669b0b7b836665d6339a34df8aa97'
URL = f'https://raw.githubusercontent.com/ymcui/cmrc2018/{REV}/squad-style-data/cmrc2018_dev.json'
SOURCE = ROOT / 'runs/cmrc2018-dev-c0eb1b6b.json'
OUT = ROOT / 'runs/cmrc2018-dev-audit-20261117.json'


def blob_sha1(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode('ascii') + data).hexdigest()


def audit(dataset):
    counts = Counter()
    seen_ids = set()
    for article in dataset['data']:
        counts['articles'] += 1
        for paragraph in article['paragraphs']:
            counts['paragraphs'] += 1
            context = paragraph['context']
            if not isinstance(context, str):
                raise ValueError('Invalid context')
            for qa in paragraph['qas']:
                counts['questions'] += 1
                question, answers, qa_id = qa['question'], qa['answers'], qa['id']
                if not isinstance(question, str) or not isinstance(answers, list) or qa_id in seen_ids:
                    raise ValueError('Invalid or duplicate question')
                seen_ids.add(qa_id)
                who = '谁' in question or '哪位' in question or '何人' in question
                counts['who_question'] += int(who)
                counts['no_answers'] += int(not answers)
                valid = 0
                for answer in answers:
                    counts['answer_annotations'] += 1
                    start, text = answer['answer_start'], answer['text']
                    if type(start) is int and isinstance(text, str) and text and start >= 0 and context[start:start + len(text)] == text:
                        valid += 1
                    else:
                        counts['invalid_offsets'] += 1
                counts['all_answers_valid'] += int(bool(answers) and valid == len(answers))
                counts['who_all_answers_valid'] += int(who and bool(answers) and valid == len(answers))
    return dict(counts)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if SOURCE.exists():
        data = SOURCE.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={'User-Agent': 'SPORA-research'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(4000000)
        if len(data) != 3299139 or blob_sha1(data) != BLOB_SHA1:
            raise ValueError('Download does not match pinned Git blob')
        with SOURCE.open('xb') as f:
            f.write(data)
    if len(data) != 3299139 or blob_sha1(data) != BLOB_SHA1:
        raise ValueError('Local source does not match pinned Git blob')
    digest = hashlib.sha256(data).hexdigest()
    counts = audit(json.loads(data.decode('utf8')))
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during audit')
    result = {'source_url': URL, 'revision': REV, 'git_blob_sha1': BLOB_SHA1, 'sha256': digest,
              'license': 'CC-BY-SA-4.0', 'counts': counts,
              'limitations': ['Question wording is a heuristic, not verified person or relation labels.',
                  'Answer offsets can be correct while answer meaning or identity is ambiguous.',
                  'This split is development data, not a sealed independent test.',
                  'No model inference, training, unanswerable examples, or memory evaluation.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
