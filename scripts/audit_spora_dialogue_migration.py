"""Compare one SPORA gate migration against its parent and the original DeepSeek."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

from audit_awake_multiturn_layers import ceval_questions, score_ceval
from audit_awake_reasoning_residual_holdout_20261121 import DATA, load_scenarios
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_open_base import SPORAGatedFeedForward
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAGeluGate, SPORAResidualBlock
import torch
from transformers import AutoTokenizer
from transformers.activations import SiLUActivation


CEVAL_GROUPS = {
    'ceval-validation-617524a': ('high_school_chinese', 'logic', 'middle_school_mathematics'),
    'ceval-native-transfer-617524a': (
        'high_school_geography', 'middle_school_history', 'middle_school_physics', 'modern_chinese_history',
    ),
}


@torch.inference_mode()
def dialogue(model, tokenizer, questions, *, open_reasoning, limit, seed):
    history, rows = [], []
    for index, question in enumerate(questions):
        history.append({'role': 'user', 'content': question})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt + ('\n' if open_reasoning else '</think>\n\n'),
                           return_tensors='pt', add_special_tokens=False).to('cuda')
        torch.manual_seed(seed + index)
        started = time.perf_counter()
        emitted = model.generate(
            **inputs, do_sample=open_reasoning,
            temperature=.6 if open_reasoning else None, top_p=.95 if open_reasoning else None,
            max_new_tokens=limit, use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        tokens = emitted[0, inputs['input_ids'].shape[1]:].tolist()
        answer = tokenizer.decode(tokens, skip_special_tokens=True)
        answer = (answer.partition('</think>')[2] if open_reasoning else answer).strip()
        rows.append({'question': question, 'answer': answer, 'tokens': tokens,
                     'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id),
                     'seconds': time.perf_counter() - started})
        history.append({'role': 'assistant', 'content': answer})
    return rows


def evaluate_dialogues(model, tokenizer, variant, *, limit, open_reasoning):
    rows = {}
    for index, scenario in enumerate(load_scenarios()):
        turns = dialogue(model, tokenizer, scenario['turns'], open_reasoning=open_reasoning,
                         limit=limit, seed=73121 + index * 100)
        rows[scenario['name']] = turns
        print(variant, scenario['name'], [len(turn['tokens']) for turn in turns],
              [bool(turn['answer']) for turn in turns], flush=True)
    return rows


def evaluate_candidate(runtime, tokenizer, ceval, *, limit, variant):
    decoder = runtime.decoder
    questions = tuple(question for folder, subjects in CEVAL_GROUPS.items()
                      for question in ceval_questions(ceval / folder, tokenizer, subjects))
    position_limit = decoder.position_table_max_position or decoder.base.config.max_position_embeddings
    academic = score_ceval(decoder, questions, position_limit)
    dialogue_rows = evaluate_dialogues(decoder, tokenizer, variant, limit=limit, open_reasoning=False)
    memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    return {'ceval': academic, 'dialogue': dialogue_rows, 'memory': memory}


def compare_reload(runtime, tokenizer, args, protected):
    reference = json.loads(args.migration_report.read_text(encoding='utf-8'))
    if reference['mode'] != 'migration' or reference['closed_limit'] != args.limit:
        raise ValueError('Reload must use the original migration settings')
    parent_sha = reference['sources'][str(args.parent)]
    if protected[args.parent] != parent_sha:
        raise ValueError('Parent checkpoint differs from the recorded migration input')
    parent, candidate = (torch.load(path, map_location='cpu', weights_only=True)
                         for path in (args.parent, args.checkpoint))
    metadata = dict(parent['metadata'])
    gates = metadata.get('gelu_gate_layers', [])
    if reference['layer'] in gates:
        raise ValueError('Migration report names a layer already migrated in the parent')
    metadata['gelu_gate_layers'] = sorted([*gates, reference['layer']])
    left, right = parent['tensors'], candidate['tensors']
    if (candidate['metadata'] != metadata or left.keys() != right.keys()
            or any(not torch.equal(tensor, right[key]) for key, tensor in left.items())):
        raise ValueError('Checkpoint differs outside the new GELU layer')
    for path, digest in reference['sources'].items():
        if protected.get(Path(path)) != digest:
            raise ValueError(f'Migration input changed: {path}')
    expected = reference['results']['single_gate_candidate']
    observed = evaluate_candidate(runtime, tokenizer, args.ceval, limit=args.limit, variant='reloaded_candidate')
    for result in (expected, observed):
        for turns in result['dialogue'].values():
            for turn in turns:
                turn.pop('seconds')
    if observed != expected:
        raise ValueError('Reload changed the migration results')
    return {'layer': reference['layer'], 'parent_sha256': parent_sha,
            'ceval_items': len(observed['ceval']),
            'dialogue_turns': sum(map(len, observed['dialogue'].values())),
            'memory_items': observed['memory']['summary']['count'],
            'identical_to_in_memory_candidate': True}


def evaluate_migration(runtime, tokenizer, args):
    decoder = runtime.decoder
    layer = decoder.base.model.layers[args.layer]
    if args.layer in decoder.layer_indices:
        layer = layer.layer
    if isinstance(layer, SPORAResidualBlock):
        layer = layer.base
    if (not isinstance(layer.mlp, SPORAGatedFeedForward)
            or not isinstance(layer.mlp.activation, (SiLUActivation, torch.nn.SiLU))):
        raise ValueError('Select a compatible layer with the original SiLU gate')
    baseline = evaluate_candidate(runtime, tokenizer, args.ceval, limit=args.limit, variant='packed_parent')
    original = layer.mlp.activation
    layer.mlp.activation = SPORAGeluGate().eval()
    try:
        candidate = evaluate_candidate(runtime, tokenizer, args.ceval, limit=args.limit,
                                        variant='single_gate_candidate')
    finally:
        layer.mlp.activation = original
    return {'packed_parent': baseline, 'single_gate_candidate': candidate}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    files = [args.checkpoint, DATA, *(MODEL / name for name in
             ('config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'))]
    if args.mode != 'raw':
        files.extend(args.ceval / folder / f'{subject}.parquet'
                     for folder, subjects in CEVAL_GROUPS.items() for subject in subjects)
    if args.mode == 'verify':
        files.extend((args.parent, args.migration_report))
    protected = {path: sha(path) for path in files}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    if args.mode == 'raw':
        results = {mode: evaluate_dialogues(base, tokenizer, mode,
                                           limit=args.open_limit if opened else args.limit,
                                           open_reasoning=opened)
                   for mode, opened in (('original_closed', False), ('original_open', True))}
    else:
        runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
        runtime.decoder.eval().requires_grad_(False)
        state_digest = frozen_digest(runtime.organism, runtime.state)
        results = (compare_reload(runtime, tokenizer, args, protected) if args.mode == 'verify'
                   else evaluate_migration(runtime, tokenizer, args))
        if frozen_digest(runtime.organism, runtime.state) != state_digest:
            raise ValueError('Candidate evaluation changed the individual')
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected input changed during evaluation')
    report = {'mode': args.mode, 'layer': args.layer if args.mode == 'migration' else None,
              'closed_limit': args.limit, 'open_limit': args.open_limit if args.mode == 'raw' else None,
              'results': results, 'seconds': time.perf_counter() - started,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'sources': {str(path): digest for path, digest in protected.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('raw', 'migration', 'verify'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, help='Directory containing both pinned C-Eval groups')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--migration-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=26)
    parser.add_argument('--limit', type=int, default=96)
    parser.add_argument('--open-limit', type=int, default=256)
    args = parser.parse_args()
    if args.mode != 'raw' and args.ceval is None:
        parser.error('migration and verify require --ceval')
    if args.mode == 'verify' and (args.parent is None or args.migration_report is None):
        parser.error('verify requires --parent and --migration-report')
    if min(args.limit, args.open_limit) < 1 or not 0 <= args.layer < 28:
        parser.error('invalid generation limit or layer index')
    main(args)
