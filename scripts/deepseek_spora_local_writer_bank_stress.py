"""Stress two feature gates on altered contexts and mixed token sequences."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_local_writer_bank_audit import ARTIFACT, OUT as BANK_REPORT, RED, SOY, SOURCE
from scripts.deepseek_spora_local_writer_stress import NAMES, TEMPLATES
from spora.research_local_writer import LocalWriterBank
from spora.semantic_text_writer import value_positions

OUT = ROOT / 'runs/deepseek-spora-local-writer-bank-stress-20261116.json'
VALUES = ('红茶', '豆浆', '绿茶', '咖啡', '红茶拿铁', '豆浆咖啡', '苹果汁')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    report = json.loads(BANK_REPORT.read_text(encoding='utf-8'))
    candidate_sha = base.sha(ARTIFACT)
    if candidate_sha != report['artifact_sha256']:
        raise ValueError('Candidate digest mismatch')
    stored = torch.load(ARTIFACT, map_location='cpu', weights_only=True)
    paths = [SOURCE / 'writer.pt', RED / 'writer.pt', SOY / 'writer.pt']
    hashes = [base.sha(path) for path in paths]
    replacements = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    if (stored['format'] != 'spora-local-writer-bank-v1' or stored['base_sha256'] != base.WEIGHT_SHA
            or stored['writer_sha256'] != hashes
            or stored['layer_sha256'] != {str(i): base.sha(path) for i, path in replacements.items()}):
        raise ValueError('Candidate source mismatch')
    weights = [torch.load(path, map_location='cpu', weights_only=True)['state']['projection.weight'] for path in paths]
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index, path in replacements.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    writer = LocalWriterBank(weights[0], weights[1:], list(stored['anchors']),
                             stored['thresholds'].tolist()).to('cuda').eval()
    target_ids = [tokenizer.encode(value, add_special_tokens=False) for value in VALUES[:2]]
    if any(len(ids) != 1 for ids in target_ids) or target_ids[0] == target_ids[1]:
        raise ValueError('Expected two distinct single-token target values')
    targets = [ids[0] for ids in target_ids]
    rows = []
    for name in NAMES:
        for value in VALUES:
            for template in TEMPLATES:
                text = template.format(name=name, value=value)
                try:
                    encoded, positions = value_positions(tokenizer, text, value)
                except ValueError:
                    rows.append({'name': name, 'value': value, 'template': template, 'status': 'unaligned'})
                    continue
                encoded = {key: tensor.to('cuda') for key, tensor in encoded.items()}
                ids = encoded['input_ids'][0, positions].tolist()
                with torch.inference_mode():
                    hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                         return_dict=True).hidden_states[1][0, positions].float()
                    matches = writer.matches(hidden).tolist()
                rows.append({'name': name, 'value': value, 'template': template, 'status': 'ok',
                             'ids': ids, 'matches': matches})
        print(json.dumps({'completed_name': name}, ensure_ascii=True), flush=True)
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(ARTIFACT) != candidate_sha
            or any(base.sha(path) != sha for path, sha in zip(paths, hashes))
            or any(base.sha(path) != stored['layer_sha256'][str(i)] for i, path in replacements.items())):
        raise ValueError('Protected source changed')
    aligned = [row for row in rows if row['status'] == 'ok']
    target_positions = [[(token, matches) for row in aligned for token, matches in zip(row['ids'], row['matches'])
                         if token == targets[index]] for index in (0, 1)]
    positions = [(token, matches) for row in aligned for token, matches in zip(row['ids'], row['matches'])]
    summary = {'cases': len(rows), 'aligned': len(aligned), 'unaligned': len(rows) - len(aligned),
               'target_position_counts': [len(group) for group in target_positions],
               'target_correct_gate': [sum(matches[index] and sum(matches) == 1 for _, matches in group)
                                       for index, group in enumerate(target_positions)],
               'wrong_gate_tokens': [sum(matches[index] and token != targets[index] for token, matches in positions)
                                     for index in (0, 1)],
               'ambiguous_tokens': sum(sum(matches) > 1 for _, matches in positions)}
    result = {'summary': summary, 'rows': rows, 'candidate_sha256': candidate_sha, 'target_ids': targets,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle value spans; no natural fact routing, memory writes, or generation.',
                              'Eight artificial names, four fixed contexts and seven selected values.',
                              'Identical target token in longer values may legitimately trigger its own gate.',
                              'Unaligned spans excluded, not counted as safe; prior development overlap is possible.',
                              'No claims about negation, factuality, identity, or longer edit sequences.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
