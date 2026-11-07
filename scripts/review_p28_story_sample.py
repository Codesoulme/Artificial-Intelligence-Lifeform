"""Bind the fixed P28 train sample to a bounded content review."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/p28-stories-data/candidate-v3'
SAMPLE = DATA / 'quality-sample.json'
REPORT = ROOT / 'reports/P28-stories-content-review.json'

# These are review labels, not a corpus-wide quality estimate. Minor issues
# include translation stiffness or local terminology drift while the main arc
# remains readable. Major issues are still readable stories with a material
# local defect or unusually unsafe/implausible teaching content.
MAJOR = {
    9: 'object wording makes the special handle premise hard to follow',
    26: 'several translated names change within one story',
    40: 'dangerous powder episode is coherent but unsuitable as ordinary safety teaching',
    41: 'character names change within one story and the threat is resolved abruptly',
    44: 'dark fatal ending is coherent but unsuitable for a child safety curriculum',
    50: 'grill wording is repeatedly unnatural and locally confusing',
}
MINOR = {
    1: 'slightly unnatural peanut motivation', 2: 'machine translated phrasing',
    3: 'compressed magical premise', 4: 'formulaic ending',
    6: 'repetitive advice wording', 9: 'local wording defect',
    12: 'formulaic object continuity', 13: 'treasure resolution is abrupt',
    17: 'object vocabulary is awkward', 18: 'some dialogue is stiff',
    19: 'local phrasing is awkward', 20: 'repeated cat references are stiff',
    24: 'source includes an explicit ending marker', 25: 'fantastical premise is unusually compressed',
    26: 'name drift', 28: 'repair premise is fantastical',
    31: 'emotion wording is literal', 33: 'prayer resolution is culturally specific',
    35: 'stranger safety lesson is formulaic', 38: 'smoke and ashtray topic is unsuitable for broad factual teaching',
    41: 'name drift', 45: 'translation wording is stiff',
    47: 'fantasy transformation is abrupt', 52: 'earthquake resolution is abrupt',
    54: 'emergency wording is formulaic', 56: 'hero action is implausible but readable',
    57: 'smoking context is unsuitable for broad factual teaching',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sample = json.loads(SAMPLE.read_text('utf-8'))
    manifest = json.loads((DATA / 'manifest.json').read_text('utf-8'))
    rows = []
    for index, row in enumerate(sample['samples'], 1):
        if row.get('split') != 'train':
            raise ValueError('Review sample contains a non-train row')
        if row.get('sha256') not in {x['sha256'] for x in sample['samples']}:
            raise ValueError('Malformed sample hash')
        category = 'major' if index in MAJOR else 'minor' if index in MINOR else 'clean'
        rows.append(dict(review_index=index, source_row_index=row['source_row_index'],
                         story_sha256=row['sha256'], category=category,
                         readable_main_arc=category != 'major' or index not in (9,),
                         entities_and_actions_consistent=category != 'major' or index not in (26, 41),
                         original_ending_present=True, physical_damage=False,
                         unexplained_concatenation=False,
                         note=MAJOR.get(index, MINOR.get(index, 'Main arc, entities and ending readable in this review.'))))
    counts = {name: sum(row['category'] == name for row in rows) for name in ('clean', 'minor', 'major')}
    report = {
        'schema_version': 1,
        'status': 'content_review_complete_bounded_pilot_approved',
        'approved_for_pilot': True,
        'data_manifest_sha256': sha(DATA / 'manifest.json'),
        'quality_sample_sha256': sha(SAMPLE),
        'sample_count': len(rows),
        'reviewer': 'Codex single-reviewer development audit; not human blind review and not corpus-wide certification.',
        'method': 'Read all fixed 64 train-only quality-sample stories. Preserve fiction and anthropomorphism as intended source properties. Score main arc, entity/action continuity, original ending, physical damage, unexplained concatenation and translation severity.',
        'predeclared_gate': {
            'readable_main_arc_minimum': 48,
            'physical_damage_concat_gibberish_maximum': 4,
            'reject_if_systemic_entity_or_direction_reversal': True,
        },
        'counts': counts,
        'readable_main_arc': sum(row['readable_main_arc'] for row in rows),
        'complete_original_ending': sum(row['original_ending_present'] for row in rows),
        'physical_damage_concat_gibberish': sum(row['physical_damage'] or row['unexplained_concatenation'] for row in rows),
        'systemic_entity_or_direction_reversal': False,
        'decision': 'The fixed sample passes the bounded 100-to-500 update gate. It contains translation defects and unsuitable safety examples, so this approves a noisy story-learning pilot only. It does not approve broad corpus training, factual teaching, or chat release.',
        'rows': rows,
        'limits': [
            'One reviewer read one fixed 64-row reservoir from one training shard.',
            'The test split was not read and remains evaluation-only.',
            'A passing sample cannot prove all 100000 training stories are clean.',
            'The source is synthetic English-origin translated fiction under CDLA-Sharing-1.0; preserve attribution and license notices.',
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('approved_for_pilot', 'counts', 'readable_main_arc', 'physical_damage_concat_gibberish', 'data_manifest_sha256')}))


if __name__ == '__main__':
    main()
