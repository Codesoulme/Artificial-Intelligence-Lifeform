"""Single-reviewer bilingual audit of previously selected no-answer examples."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.audit_chinese_squadv2_20261117 import SOURCE, SHA
from scripts.deepseek_squadv2_who_probe_20261117 import OUT as PRIOR
from scripts.select_chinese_squadv2_who_20261117 import OUT as MANIFEST

OUT = ROOT / 'runs/review-squadv2-noanswer-20261117.json'
REVIEW = {
    '06d46287f9817f6b5e14245028345a78fb0987d3052e15ae0b0a2cbac5b31c96': ('ambiguous', 'geographic scope may encompass mentioned campaign'),
    '04b8744bad2c0c3a64709a04ad182073d150b709ef9d1651070f3ae69a747bc1': ('supported', 'requested material differs from stated material'),
    '0606d963ff179d1263d4623daf23809d2913fde7c8124e66d965b1df663086d0': ('supported', 'current officeholder does not establish first'),
    '07096549c051f89b1ebf6ac8b7d4ef614406f483de8a594efbb49a6b6b9bf381': ('supported', 'visit does not establish ownership'),
    '00c438ffdd74adbd017d7dd066e1bf53deabc74019f25f07a5d8b7437c1f4e95': ('supported', 'another person is not linked to requested visit'),
    '009602b16f31fcd7e75feeae7e531f1cdc145d8c930d71a6f289780673ac6c44': ('supported', 'related theory does not establish requested discovery'),
    '026af353cd43bd23d81830c1c4773e4f62034e08daa96a0b06403b8adbf4dfde': ('supported', 'often cooperating does not specify never cooperating'),
    '042c2e5733ee809981da09a2d4fd693b550ee976b9e8440936af01063ca318c5': ('ambiguous', 'broad responsibility could be read as covering requested action'),
    '027a990dc597f0aaa3f96d74ff6d0d0444efa9c65b39c34d072ba32f0d56af80': ('ambiguous', 'the requested zoning violation may be read as a code violation'),
    '065878f6c86727cefbc5a26045769328355704a7e68fa2ad4b761a50284e7b08': ('supported', 'attacker and attacked location differ'),
    '05733b76fe8620f0bf221507ed237c3605130741b1f5e7a16c93eef86ad8eaeb': ('ambiguous', 'travelling with may be interpreted as staying with'),
    '03852d3f55a556540d8ccea5160e445d06ed4927b78ddafc53245d54fd7fdf80': ('ambiguous', 'contribution and reporting relations are distinguishable but could be inferred'),
    '0163cb8f6a6b089b6458e3c226776338d3288774125bba861bbea2f459fc369a': ('supported', 'the calibration direction is reversed in question'),
    '01fe46809d87604e690f4f8092b3d23ff3f83371c3e6e977579c74fcb82bba84': ('ambiguous', 'asked balance relation partly present but modifier contradicts passage'),
    '050d4f073429024b645604a535b9dc7279fb49311ca5d0a904e98134016878c9': ('supported', 'requested office differs from the one mentioned'),
    '0204129e5f6054c99ecd12dc17c3041ce200478f55156f660e00c47196dc9652': ('ambiguous', 'negotiating surrender may be broadly read as wartime negotiation'),
}


def summarize(selection, prior, review):
    ids = [r['digest'] for r in selection['rows'] if r['kind'] == 'unknown']
    predictions = {r['digest']: r for r in prior['rows'] if r['group'] == 'unknown'}
    if len(ids) != len(set(ids)) or set(ids) != set(review) or set(ids) != set(predictions):
        raise ValueError('Review manifest mismatch')
    result = {}
    for status in ('supported', 'ambiguous'):
        subset = [predictions[d] for d in ids if review[d][0] == status]
        result[status] = {'total': len(subset), **{name: sum(r['heads'][name]['selected'] for r in subset)
                          for name in ('prior', 'natural', 'question')}}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source mismatch')
    selection = json.loads(MANIFEST.read_text(encoding='utf8'))
    prior = json.loads(PRIOR.read_text(encoding='utf8'))
    if selection['source_sha256'] != SHA or prior['source_sha256']['data'] != SHA:
        raise ValueError('Selection source mismatch')
    result = {'summary': summarize(selection, prior, REVIEW), 'categories': dict(Counter(v[0] for v in REVIEW.values())),
              'source_sha256': SHA, 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              'prior_sha256': hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
              'reviewer': 'one model compared Chinese and English context and questions after seeing predictions',
              'limitations': ['Review is subjective and not independent or blinded.',
                  'Supported means evidence for absent answer under strict reading, not guaranteed correctness.',
                  'No review of answerable cases; no model inference or real memory evaluation.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
