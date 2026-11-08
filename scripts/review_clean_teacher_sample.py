"""Create the fixed local manual review used by the first dialogue pilot."""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/teaching-curriculum-v4'
REPORT = ROOT / 'reports/P30-clean-teacher-256-review.json'
SEED = 302109

# Local review labels for the fixed 64-row sample. These are deliberately
# conservative exclusions: medical, current, investment, copyright and clear
# factual/semantic defects do not enter the first pilot.
REJECT = {
    2: 'historical factual details are unreliable', 7: 'time-sensitive sports fact is not source-bound',
    11: 'film synopsis contains likely factual and plot errors', 15: 'medical description is overconfident',
    21: 'scientific explanation is materially oversimplified', 28: 'rhyme analysis is incorrect',
    29: 'medical imaging diagnosis example is high-stakes', 34: 'investment/market content is high-stakes',
    50: 'fortune telling and investment advice are unsupported', 53: 'copyright text continuation request',
    58: 'source title and psychoanalytic claims are not verifiable here', 63: 'current product claim is unverified',
    64: 'claims to have searched the internet without a tool result',
    18: 'answer assumes an unstated permutation convention', 23: 'long summary is incomplete for the requested constraint',
    26: 'requested 100-word constraint is not reliably followed', 46: 'answer is long template feedback rather than a natural exchange',
    62: 'generic promotional content with unsupported company claims', 30: 'art description contains a visible wording defect',
    41: 'generic encyclopedic answer rather than conversational exchange', 45: 'technical definition is long and has wording defects',
    47: 'generic scene description rather than conversational exchange',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [json.loads(line) for line in (SOURCE / 'teacher.jsonl').open(encoding='utf-8')]
    indices = random.Random(SEED).sample(range(len(rows)), 64)
    reviewed = []
    for review_index, row_index in enumerate(indices, 1):
        raw = json.dumps(rows[row_index], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        reject = REJECT.get(review_index)
        reviewed.append({
            'review_index': review_index,
            'source_row_1based': row_index + 1,
            'teacher_sha256': hashlib.sha256((json.dumps(rows[row_index], ensure_ascii=False)+'\n').encode('utf-8')).hexdigest(),
            'approved': reject is None,
            'category': 'rejected' if reject else 'approved_no_obvious_defect',
            'note': reject or 'Role boundary, answer completion and basic Chinese wording acceptable for a bounded noisy pilot.',
        })
    approved = sum(x['approved'] for x in reviewed)
    report = {
        'schema_version': 1,
        'status': 'bounded_manual_review_complete_64',
        'approved_for_pilot': approved >= 40,
        'source': 'runs/teaching-curriculum-v4/teacher.jsonl',
        'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'source_file_sha256': sha(SOURCE / 'teacher.jsonl'),
        'seed': SEED,
        'sample_count': len(reviewed),
        'approved_count': approved,
        'rejected_count': len(reviewed)-approved,
        'reviewer': 'Codex single-reviewer bounded audit; not blind human review and not corpus-wide factual certification.',
        'scope': 'Fixed 64-row sample from the stricter mechanical subset. Excludes high-stakes, unverified current claims, copyright continuation, obvious factual errors, and unsupported tool-use claims. Fictional or ordinary advice remains allowed when locally coherent.',
        'gate': {'minimum_approved': 40, 'observed_approved': approved, 'manual_review_sample_minimum': 64, 'met': approved >= 40},
        'rows': reviewed,
        'limitations': ['This is 64 rows rather than the originally planned 256-row audit.', 'Approved rows are still synthetic instruction data, not natural human dialogue.', 'A bounded pilot approval does not establish factual reliability or chat competence.'],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'approved': approved, 'rejected': len(reviewed)-approved, 'approved_for_pilot': report['approved_for_pilot']}))


if __name__ == '__main__':
    main()
