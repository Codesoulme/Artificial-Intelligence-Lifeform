"""Read-only development stress test for trained fact-conditioned span head."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT
from scripts.deepseek_spora_general_subject_locator import extract
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-crossed-stress-20261117.json'
CANDIDATE = ROOT / 'runs/deepseek-fact-crossed-names8-1200-20261117/head.pt'
PAIRS = (('小庄', '阿齐'), ('小芮', '阿谭'), ('小霜', '阿岳'), ('小岚', '阿祺'))


def cases():
    groups = {'paraphrase': [], 'same_type': [], 'homonym': []}
    for a, b in PAIRS:
        for reverse in (False, True):
            p, q = (b, a) if reverse else (a, b)
            examples = [
                ('paraphrase', f'{p}喜欢喝茶，{q}喜欢蓝色。',
                 (('喜欢茶的人是谁？', p), ('喜欢蓝色的人是谁？', q), ('喜欢跑步的人是谁？', None))),
                ('same_type', f'{p}喜欢喝茶，{q}喜欢喝咖啡。',
                 (('谁喜欢喝茶？', p), ('谁喜欢喝咖啡？', q), ('谁喜欢喝果汁？', None))),
                ('homonym', f'有两个人都叫{p}，其中一个喜欢喝茶。',
                 (('谁喜欢喝茶？', None),)),
            ]
            for group, evidence, queries in examples:
                for query, target in queries:
                    text = evidence + query
                    left = text.index(target) if target else None
                    groups[group].append({'text': text, 'kind': 'known' if target else 'unknown',
                                          'label': 1 if target else None, 'left': left,
                                          'right': left + len(target) if target else None})
    return groups


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'before': CONTEXT / 'head.pt', 'report': CONTEXT / 'result.json',
             'candidate': CANDIDATE}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    prior = torch.load(paths['before'], map_location='cpu', weights_only=True)
    candidate = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['before'] != report['head_sha256']
            or any(prior['source_sha256'][k] != hashes[k] for k in source_paths())
            or any(candidate['source_sha256'][k] != hashes[k] for k in source_paths())
            or candidate['source_sha256']['context'] != hashes['before']):
        raise ValueError('Source mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-fact-stress-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        data = {name: extract(chat, rows) for name, rows in cases().items()}
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = data['paraphrase'][0]['features'].shape[-1]
    heads = {'before': ContextSpanSelector(width, rank=prior['rank']),
             'candidate': ContextSpanSelector(width, rank=candidate['rank'])}
    heads['before'].load_state_dict(prior['state'], strict=True)
    heads['candidate'].load_state_dict(candidate['state'], strict=True)
    results = {name: {group: evaluate(head, items) for group, items in data.items()}
               for name, head in heads.items()}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    output = {'results': results, 'source_sha256': hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Authored small dependent examples; development only.',
                  'Homonym labels require identity discrimination beyond mention selection.',
                  'No natural conversation, actual memory reads, writes, or production regression.',
                  'Candidate file hash recorded now; training report did not include its hash.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
