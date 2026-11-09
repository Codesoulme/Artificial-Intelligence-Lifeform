"""Infer conservative sequence groups from chapter markers; never trains."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'runs/p192-chinese-vietnamese-literature-20261109/dataset_zh_vi.parquet'
EXPECTED_SHA = '2676db2aabdb3d6b5c2618b7cc8fa9ad283ae4e80bb0c8366065134295ce3b4e'
MARKER = re.compile(r'^\s*第\s*([0-9零〇一二两三四五六七八九十百千万]+)\s*(回|章)')
DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '百': 100, '千': 1000, '万': 10000}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chinese_number(value):
    if value.isdigit():
        return int(value)
    total = current = 0
    for char in value:
        n = DIGITS.get(char)
        if n is None:
            return None
        if n >= 10:
            total += (current or 1) * n
            current = 0
        else:
            current = current * 10 + n
    return total + current


def infer_groups(zh):
    markers = []
    for index, text in enumerate(zh):
        match = MARKER.match((text or '').strip())
        if match:
            number = chinese_number(match.group(1))
            if number is not None:
                heading = (text or '').strip()
                tail = heading[match.end():]
                flags = []
                if tail and not tail[0].isspace() and tail[0] not in ':：':
                    flags.append('no_heading_separator')
                if len(heading) > 80:
                    flags.append('long_heading_may_include_prose')
                markers.append({'row': index, 'number': number, 'unit': match.group(2), 'heading': heading[:200], 'flags': flags})
    groups = []
    current = None
    for marker in markers:
        reset = current is None or marker['unit'] != current['unit'] or marker['number'] <= current['last_number']
        if reset:
            if current is not None:
                current['end_marker_row'] = marker['row'] - 1
                groups.append(current)
            current = {'group_id': f'g{len(groups):04d}', 'start_marker_row': marker['row'], 'unit': marker['unit'], 'first_number': marker['number'], 'last_number': marker['number'], 'marker_count': 1, 'headings': [marker['heading']], 'last_heading': marker['heading'], 'number_gaps': [], 'flagged_marker_count': bool(marker['flags'])}
        else:
            gap = marker['number'] - current['last_number']
            if gap > 1:
                current['number_gaps'].append({'after': current['last_number'], 'before': marker['number'], 'row': marker['row']})
            current['last_number'] = marker['number']
            current['last_heading'] = marker['heading']
            current['marker_count'] += 1
            current['flagged_marker_count'] += bool(marker['flags'])
            if len(current['headings']) < 3:
                current['headings'].append(marker['heading'])
    if current is not None:
        current['end_marker_row'] = len(zh) - 1
        groups.append(current)
    for group in groups:
        start, end = group['start_marker_row'], group['end_marker_row']
        group['row_count'] = end - start + 1
        group['characters'] = sum(len(text or '') for text in zh[start:end + 1])
        group['review_flags'] = ['unverified_work_boundary', 'unverified_rights']
        if group['marker_count'] == 1:
            group['review_flags'].append('single_marker')
        if group['first_number'] != 1:
            group['review_flags'].append('starts_after_first_chapter')
        if group['number_gaps']:
            group['review_flags'].append('chapter_number_gaps')
        if group['flagged_marker_count']:
            group['review_flags'].append('possible_prose_or_heading_artifacts')
    return markers, groups


def infer(source, output):
    if output.exists():
        raise FileExistsError(output)
    if sha(source) != EXPECTED_SHA:
        raise ValueError('source hash mismatch')
    zh = pq.read_table(source, columns=['zh']).column('zh').to_pylist()
    markers, groups = infer_groups(zh)
    result = {'status': 'sequence_boundary_inference_not_training_approval', 'training_approval': False, 'source_sha256': sha(source), 'rows': len(zh), 'chapter_markers': len(markers), 'inferred_groups': len(groups), 'unassigned_prefix_rows': groups[0]['start_marker_row'] if groups else len(zh), 'groups': groups, 'method': 'start-of-row 第N回/章 markers; new group on unit change or non-increasing chapter number; flags do not verify headings', 'limitations': ['Inferred groups are heuristic and may split novels, merge works, or miss unnumbered boundaries.', 'A marker can be prose and a group is not a work or a valid split unit.', 'No explicit work ID exists in the source schema.', 'Groups require manual/source-level validation before train/dev splitting.', 'No training authorization.']}
    output.mkdir(parents=False)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    with (output / 'groups.jsonl').open('x', encoding='utf8', newline='\n') as stream:
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False, separators=(',', ':')) + '\n')
    with (output / 'markers.jsonl').open('x', encoding='utf8', newline='\n') as stream:
        for marker in markers:
            stream.write(json.dumps(marker, ensure_ascii=False, separators=(',', ':')) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = infer(args.source, args.output)
    print(json.dumps({k: result[k] for k in ('output' if 'output' in result else 'status', 'rows', 'chapter_markers', 'inferred_groups', 'training_approval')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
