"""Inspect two pinned MemoryBench training subsets, without evaluating a model."""
import ast
from collections import Counter
import hashlib
import json
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/research/memorybench-train-source-20261127-v1'
COMMIT = '3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d'
CONFIGS = ('JuDGE', 'WritingBench-Creative&Design')


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'SPORA-research'}), timeout=60) as response:
        return response.read()


def parse(value):
    if not isinstance(value, str): return value
    if len(value) > 2_000_000: return None
    try: return json.loads(value)
    except (ValueError, TypeError):
        try: return ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError): return None


def inspect(path):
    with path.open('rb') as source:
        table = ipc.open_stream(source).read_all()
    result = {'rows': table.num_rows, 'columns': table.column_names, 'languages': Counter(),
              'dataset_names': Counter(), 'dialog_parse_type': Counter(), 'dialog_lengths': Counter(),
              'dialog_roles': Counter(), 'feedback_parse_type': Counter(), 'feedback_keys': Counter(),
              'feedback_entries': 0, 'dialog_entries': 0, 'empty_dialog': 0, 'empty_feedback': 0}
    for row in table.to_pylist():
        result['languages'][str(row.get('lang'))] += 1
        result['dataset_names'][str(row.get('dataset_name'))] += 1
        dialog, feedback = parse(row.get('dialog')), parse(row.get('implicit_feedback'))
        result['dialog_parse_type'][type(dialog).__name__] += 1
        result['feedback_parse_type'][type(feedback).__name__] += 1
        if isinstance(dialog, list):
            result['dialog_lengths'][str(len(dialog))] += 1
            result['dialog_entries'] += len(dialog)
            result['empty_dialog'] += not dialog
            for turn in dialog:
                if isinstance(turn, dict): result['dialog_roles'][str(turn.get('role'))] += 1
        if isinstance(feedback, list):
            result['feedback_entries'] += len(feedback)
            result['empty_feedback'] += not feedback
            for item in feedback:
                if isinstance(item, dict): result['feedback_keys'].update(item.keys())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def stage(name):
        item = {'stage': name, 'seconds': time.perf_counter() - start}
        (OUT / 'stage.json').write_text(json.dumps(item), encoding='utf8')
        print(json.dumps(item), flush=True)
    try:
        files = {}
        for i, config in enumerate(CONFIGS):
            stage('metadata_' + str(i))
            parent = 'dataset/' + config + '/train'
            quoted = urllib.parse.quote(parent, safe='/')
            api = 'https://huggingface.co/api/datasets/THUIR/MemoryBench/tree/' + COMMIT + '/' + quoted + '?expand=true'
            entries = json.loads(get(api))
            arrow = [e for e in entries if e.get('path', '').endswith('.arrow')]
            if len(arrow) != 1 or not arrow[0].get('lfs', {}).get('oid'):
                raise ValueError('Expected exactly one LFS Arrow file')
            entry = arrow[0]
            url = 'https://huggingface.co/datasets/THUIR/MemoryBench/resolve/' + COMMIT + '/' + urllib.parse.quote(entry['path'], safe='/')
            stage('download_' + str(i))
            content = get(url)
            digest = hashlib.sha256(content).hexdigest()
            if len(content) != entry['size'] or digest != entry['lfs']['oid']:
                raise ValueError('LFS content mismatch')
            path = OUT / ('train-' + str(i) + '.arrow')
            with path.open('xb') as stream: stream.write(content)
            stage('inspect_' + str(i))
            files[config] = {'repository_path': entry['path'], 'size': len(content), 'sha256': digest,
                             'counts': inspect(path)}
        result = {'commit': COMMIT, 'files': files, 'seconds': time.perf_counter() - start}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
