"""Compare conservative paragraph extraction on the existing P225 review sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.wikibooks_structured import extract

SAMPLE_SHA = 'd56e0b6bb4eae04dd0d2b2cee75c975900b95e7271778d4a310c6a1256277280'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(sample, output):
    sample, output = Path(sample), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = sample.read_bytes()
    if digest(raw) != SAMPLE_SHA:
        raise ValueError('Review sample digest mismatch')
    rows, counts, risks = [], Counter(), Counter()
    for index, item in enumerate(json.loads(raw)):
        text = item['raw_wikitext']
        if digest(text.encode('utf8')) != item['raw_sha256']:
            raise ValueError('Source page digest mismatch')
        result = extract(text)
        paragraphs = result['paragraphs']
        accepted = [p for p in paragraphs if p['status'] == 'review_candidate']
        quarantined = [p for p in paragraphs if p['status'] == 'quarantined']
        if any(text[p['start']:p['end']] != p['raw'] for p in paragraphs):
            raise ValueError('Paragraph offset mismatch')
        for p in paragraphs:
            if any(text[r['start']:r['end']] != r['raw'] for r in p['risks']):
                raise ValueError('Risk offset mismatch')
        reasons = Counter(r['reason'] for p in quarantined for r in p['risks'])
        risks.update(reasons)
        counts['pages'] += 1
        counts['paragraphs'] += len(paragraphs)
        counts['review_candidate_paragraphs'] += len(accepted)
        counts['quarantined_paragraphs'] += len(quarantined)
        counts['review_candidate_characters'] += sum(len(p['text']) for p in accepted)
        counts['pages_with_candidate'] += bool(accepted)
        rows.append({'index': index, 'page_id': item['page_id'], 'raw_sha256': item['raw_sha256'],
                     'paragraphs': len(paragraphs), 'review_candidate_paragraphs': len(accepted),
                     'quarantined_paragraphs': len(quarantined), 'risk_reasons': dict(reasons)})
    result = {'status': 'development_extraction_audit_not_training_approval',
              'sample_sha256': SAMPLE_SHA, 'extractor_sha256': digest(Path(__file__).with_name('wikibooks_structured.py').read_bytes()),
              'counts': dict(counts), 'risk_occurrences': dict(risks), 'pages': rows,
              'limitations': ['Candidate paragraphs may depend on omitted surrounding context.',
                              'Risk occurrence counts overlap and do not measure quality.',
                              'This development sample informed the extraction design and is not independent validation.',
                              'No factual, rights, privacy, deduplication or training approval is implied.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.sample, args.output)
    print(json.dumps({'counts': result['counts'], 'risk_occurrences': result['risk_occurrences']}))


if __name__ == '__main__':
    main()
