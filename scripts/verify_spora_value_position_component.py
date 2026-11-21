"""Pack the trained value head and verify it after loading in a new process."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_ceval
from audit_awake_reasoning_residual_holdout_20261121 import DATA, load_scenarios
from audit_spora_dialogue_migration import CEVAL_GROUPS
from audit_spora_split_position_attention import compare_generation_cache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.pretrained_attention import load_value_position_attention
from train_spora_grouped_absolute_attention import LAYER, MAX_POSITION, checkpoint_baseline, evaluate_candidate


def write_report(path, report):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def pack(args):
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    source_digest = sha(args.checkpoint)
    attention_digest = sha(args.attention)
    for path, digest in ((args.checkpoint, source_digest), (args.attention, attention_digest)):
        if audit['sources'].get(str(path)) != digest:
            raise ValueError(f'Component experiment used a different input: {path}')
    value = audit['results']['value_only']
    summary = value['summary']
    if (value['groups'] or len(value['value_groups']) != 1
            or not all(value['generation_cache'].values())
            or summary['ceval_after'] < summary['ceval_before']
            or summary['memory_after'] != summary['memory_before']):
        raise ValueError('Value-only experiment did not meet the component gate')
    group = value['value_groups'][0]
    source = torch.load(args.attention, map_location='cpu', weights_only=True)
    config = json.loads((MODEL / 'config.json').read_text(encoding='utf-8'))
    width = config['hidden_size']
    key_heads = config['num_key_value_heads']
    head_dim = width // config['num_attention_heads']
    state = source['state']
    start = width + (key_heads + group) * head_dim
    end = start + head_dim
    args.output.mkdir(parents=True, exist_ok=False)
    component = args.output / 'value-position.pt'
    atomic_save({'format': 'spora-value-position-component-v1', 'layer': LAYER, 'group': group,
                 'source_sha256': source_digest, 'position': state['position.weight'],
                 'weight': state['qkv.weight'][start:end].clone(), 'bias': state['qkv.bias'][start:end].clone()},
                component)
    write_report(args.output / 'packed.json', {'component_sha256': sha(component), 'source_sha256': source_digest,
                                              'audit_sha256': sha(args.audit), 'group': group})
    print('Packed', component, flush=True)


def verify(args):
    component = args.output / 'value-position.pt'
    destination = args.output / 'verified.json'
    if destination.exists():
        raise FileExistsError(destination)
    packed = json.loads((args.output / 'packed.json').read_text(encoding='utf-8'))
    extra_folder, extra_subjects = next(
        (folder, subjects) for folder, subjects in CEVAL_GROUPS.items() if subjects != CEVAL_SUBJECTS
    )
    extra_path = args.ceval.parent / extra_folder
    sources = (args.checkpoint, args.baseline, args.reload_report, args.audit, component, DATA,
               *(MODEL / name for name in ('model.safetensors', 'tokenizer.json', 'tokenizer_config.json')),
               *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS),
               *(extra_path / f'{subject}.parquet' for subject in extra_subjects))
    protected = {path: sha(path) for path in sources}
    if protected[args.checkpoint] != packed['source_sha256'] or protected[args.audit] != packed['audit_sha256']:
        raise ValueError('Parent checkpoint or component evaluation differs from the packed manifest')
    baseline = checkpoint_baseline(args, protected)
    migration = json.loads(args.baseline.read_text(encoding='utf-8'))
    for subject in extra_subjects:
        path = extra_path / f'{subject}.parquet'
        if migration['sources'].get(str(path)) != protected[path]:
            raise ValueError(f'Mismatched additional C-Eval input: {path}')
    original_audit = json.loads(args.audit.read_text(encoding='utf-8'))['results']['value_only']
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    state = frozen_digest(runtime.organism, runtime.state)
    layer = decoder.base.model.layers[LAYER]
    layer.self_attn = load_value_position_attention(component, layer.self_attn,
                                                   checkpoint_digest=protected[args.checkpoint],
                                                   component_digest=packed['component_sha256'])
    result = evaluate_candidate(runtime, tokenizer, args.ceval, baseline)
    questions = tuple(ceval_questions(extra_path, tokenizer, extra_subjects))
    additional = score_ceval(decoder, questions, MAX_POSITION)
    prior = [row for row in baseline['ceval'] if row['subject'] in extra_subjects]
    reference_eos = sum(turn['eos'] for rows in baseline['dialogue'].values() for turn in rows)
    candidate_eos = sum(turn['eos'] for rows in result['dialogue'].values() for turn in rows)
    ordinary = closed_think_inputs(tokenizer, load_scenarios()[1]['turns'][0], 'cuda')
    cache = compare_generation_cache(runtime, tokenizer, ordinary)
    exact = {
        'ceval': result['ceval'] == original_audit['ceval'],
        'dialogue_turns': sum(left['tokens'] == right['tokens'] for name, rows in result['dialogue'].items()
                              for left, right in zip(rows, original_audit['dialogue'][name], strict=True)),
        'memory': result['memory'] == original_audit['memory'],
    }
    summary = result['summary']
    added_before = sum(row['choice'] == row['expected'] for row in prior)
    added_after = sum(row['choice'] == row['expected'] for row in additional)
    accepted = (summary['ceval_after'] >= summary['ceval_before'] and added_after >= added_before
                and summary['memory_after'] == summary['memory_before'] and candidate_eos >= reference_eos
                and all(cache.values()))
    if frozen_digest(runtime.organism, runtime.state) != state or any(
        sha(path) != digest for path, digest in protected.items()
    ):
        raise ValueError('A protected model or input changed during the audit')
    report = {'accepted': accepted, 'result': result, 'matches_split_value': exact,
              'additional_ceval': additional, 'additional_before': added_before, 'additional_after': added_after,
              'dialogue_eos_before': reference_eos, 'dialogue_eos_after': candidate_eos,
              'generation_cache': cache, 'seconds': time.perf_counter() - started,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'sources': {str(path): digest for path, digest in protected.items()}}
    write_report(destination, report)
    print('Extension checked', exact, summary, added_before, added_after, 'accepted', accepted, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pack', 'verify'))
    for name in ('checkpoint', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('attention', 'baseline', 'reload-report', 'ceval'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.mode == 'pack':
        if args.attention is None:
            parser.error('pack requires --attention')
        pack(args)
    else:
        if any(path is None for path in (args.baseline, args.reload_report, args.ceval)):
            parser.error('verify requires --baseline, --reload-report and --ceval')
        verify(args)
