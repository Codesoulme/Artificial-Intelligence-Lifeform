"""Compare independent blinded subagent review with earlier single-model audit."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.audit_chinese_squadv2_20261117 import SOURCE, SHA
from scripts.review_squadv2_noanswer_20261117 import REVIEW, summarize
from scripts.select_chinese_squadv2_who_20261117 import OUT as MANIFEST
from scripts.deepseek_squadv2_who_probe_20261117 import OUT as PRIOR

OUT = ROOT / 'runs/compare-squadv2-independent-review-20261117.json'
INDEPENDENT = {
    '009602b16f31fcd7e75feeae7e531f1cdc145d8c930d71a6f289780673ac6c44': 'supported',
    '00c438ffdd74adbd017d7dd066e1bf53deabc74019f25f07a5d8b7437c1f4e95': 'supported',
    '0163cb8f6a6b089b6458e3c226776338d3288774125bba861bbea2f459fc369a': 'supported',
    '01fe46809d87604e690f4f8092b3d23ff3f83371c3e6e977579c74fcb82bba84': 'supported',
    '0204129e5f6054c99ecd12dc17c3041ce200478f55156f660e00c47196dc9652': 'ambiguous',
    '026af353cd43bd23d81830c1c4773e4f62034e08daa96a0b06403b8adbf4dfde': 'supported',
    '027a990dc597f0aaa3f96d74ff6d0d0444efa9c65b39c34d072ba32f0d56af80': 'ambiguous',
    '03852d3f55a556540d8ccea5160e445d06ed4927b78ddafc53245d54fd7fdf80': 'ambiguous',
    '042c2e5733ee809981da09a2d4fd693b550ee976b9e8440936af01063ca318c5': 'ambiguous',
    '04b8744bad2c0c3a64709a04ad182073d150b709ef9d1651070f3ae69a747bc1': 'supported',
    '050d4f073429024b645604a535b9dc7279fb49311ca5d0a904e98134016878c9': 'supported',
    '05733b76fe8620f0bf221507ed237c3605130741b1f5e7a16c93eef86ad8eaeb': 'ambiguous',
    '0606d963ff179d1263d4623daf23809d2913fde7c8124e66d965b1df663086d0': 'supported',
    '065878f6c86727cefbc5a26045769328355704a7e68fa2ad4b761a50284e7b08': 'supported',
    '06d46287f9817f6b5e14245028345a78fb0987d3052e15ae0b0a2cbac5b31c96': 'ambiguous',
    '07096549c051f89b1ebf6ac8b7d4ef614406f483de8a594efbb49a6b6b9bf381': 'supported',
}


def compare(selection, prior):
    ids = {row['digest'] for row in selection['rows'] if row['kind'] == 'unknown'}
    if ids != set(REVIEW) or ids != set(INDEPENDENT):
        raise ValueError('Review manifest mismatch')
    consensus = {key: ('supported' if REVIEW[key][0] == INDEPENDENT[key] == 'supported' else 'ambiguous', '')
                 for key in ids}
    return {'agreement': sum(REVIEW[key][0] == INDEPENDENT[key] for key in ids),
            'total': len(ids), 'conservative': summarize(selection, prior, consensus)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source mismatch')
    selection = json.loads(MANIFEST.read_text(encoding='utf8'))
    prior = json.loads(PRIOR.read_text(encoding='utf8'))
    if selection['source_sha256'] != SHA or prior['source_sha256']['data'] != SHA:
        raise ValueError('Selection source mismatch')
    result = {'summary': compare(selection, prior), 'source_sha256': SHA,
              'limitations': ['Two model reviewers, not two independent human annotators.',
                  'First reviewer saw predictions; second was instructed not to inspect them.',
                  'Labels are subjective; no independently adjudicated gold.',
                  'The 16 selected development examples remain previously inspected.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
