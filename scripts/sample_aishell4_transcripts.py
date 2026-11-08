"""Build a local-only audit of a pinned AISHELL-4 TextGrid sample."""
import collections
import hashlib
import json
import random
from pathlib import Path

from praatio import textgrid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'runs/research/aishell4/sample-v1'
OUTPUT = DATA/'transcript-audit.json'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    rng = random.Random(811109)
    scenes, totals = [], collections.Counter()
    for file_index, item in enumerate(manifest['files']):
        path = DATA/item['local_file']
        if sha(path) != item['sha256']:
            raise ValueError('Source changed')
        grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
        segments = []
        for speaker, name in enumerate(grid.tierNames, 1):
            tier = grid.getTier(name)
            for interval in tier.entries:
                value = interval.label.strip()
                if not value:
                    continue
                if interval.start > interval.end:
                    raise ValueError('Invalid interval')
                segments.append(dict(start=interval.start,end=interval.end,speaker=speaker,text=value))
                totals['segments'] += 1
                totals['characters'] += len(value)
                totals['han_characters'] += sum('\u4e00' <= c <= '\u9fff' for c in value)
        segments.sort(key=lambda x:(x['start'],x['end'],x['speaker']))
        running_end = -1.0
        for row in segments:
            if row['start'] < running_end:
                totals['segments_overlapping_prior'] += 1
            running_end = max(running_end,row['end'])
        totals['files'] += 1
        totals['speaker_tiers'] += len(grid.tierNames)
        if len(segments) < 8:
            raise ValueError('Not enough intervals for audit')
        for start in rng.sample(range(len(segments)-7),2):
            # Speaker numbering is local to each file; no metadata is exported.
            scenes.append(dict(file_index=file_index, offset=start, segments=segments[start:start+8]))
    report = dict(status='audit_sample_not_approved_for_training', seed=811109,
                  source_manifest_sha256=sha(DATA/'manifest.json'), evaluator_sha256=sha(Path(__file__)),
                  totals=totals, sample_scenes=len(scenes), scenes=scenes,
                  scope='12 sampled train recordings; order by interval start, not proof of clean turn boundaries. Overlapping speech and personal information may remain. No model training.')
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'scenes'},ensure_ascii=False))


if __name__ == '__main__':
    main()
