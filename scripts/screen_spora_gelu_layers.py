"""Screen late feed-forward gate changes against a shared checkpoint baseline."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from scripts.audit_local_named_chat import run
from scripts.audit_named_dialogue_midlayers_holdout import CASES
from scripts.audit_spora_dialogue_migration import CEVAL_GROUPS, DATA, evaluate_candidate
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_open_base import SPORAGatedFeedForward
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAGeluGate, SPORAResidualBlock
from transformers import AutoTokenizer
from transformers.activations import SiLUActivation


def compare(parent, candidate):
    def academic(rows):
        return sum(row['choice'] == row['expected'] for row in rows)

    original_turns = [turn for turns in parent['dialogue'].values() for turn in turns]
    changed_turns = [turn for turns in candidate['dialogue'].values() for turn in turns]
    short = candidate['short']
    return {
        'ceval': academic(candidate['ceval']),
        'changed_ceval': sum(a['choice'] != b['choice']
                             for a, b in zip(parent['ceval'], candidate['ceval'], strict=True)),
        'identical_open_turns': sum(a['tokens'] == b['tokens']
                                    for a, b in zip(original_turns, changed_turns, strict=True)),
        'open_eos': sum(turn['eos'] for turn in changed_turns),
        'short_cases': short['cases_passed'],
        'short_turns': short['turns_passed'],
        'memory_exact': candidate['memory']['summary']['exact'],
        'memory_identical': candidate['memory'] == parent['memory'],
    }


def evaluate(runtime, tokenizer, ceval, name):
    result = evaluate_candidate(runtime, tokenizer, ceval, limit=96, variant=name)
    result['short'] = run(runtime.decoder, tokenizer, cases=CASES)
    return result


def inputs(checkpoint, ceval):
    sources = [checkpoint, DATA, *(MODEL / name for name in
               ('config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'))]
    sources.extend(ceval / folder / f'{subject}.parquet' for folder, subjects in CEVAL_GROUPS.items()
                   for subject in subjects)
    return {str(path): sha(path) for path in sources}


def load(checkpoint):
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    return runtime, tokenizer


def verify(output, checkpoint, ceval, protected):
    report = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    candidate_path = output / 'candidate.pt'
    digest = sha(candidate_path)
    if report['sources'] != protected or digest != report['candidate_sha256']:
        raise ValueError('A migration source changed before reload')
    original, candidate = (torch.load(path, map_location='cpu', weights_only=True)
                           for path in (checkpoint, candidate_path))
    metadata = dict(original['metadata'])
    metadata['gelu_gate_layers'] = sorted([*metadata.get('gelu_gate_layers', ()), *report['accepted']])
    if (candidate['metadata'] != metadata or original['tensors'].keys() != candidate['tensors'].keys()
            or any(tensor.dtype != candidate['tensors'][key].dtype or not torch.equal(tensor, candidate['tensors'][key])
                   for key, tensor in original['tensors'].items())):
        raise ValueError('The saved candidate changed more than the accepted gate configuration')
    del original, candidate
    runtime, tokenizer = load(candidate_path)
    observed = evaluate(runtime, tokenizer, ceval, 'reloaded')
    reference = next(trial['evaluation'] for trial in reversed(report['trials']) if trial['accepted'])
    for result in (observed, reference):
        for turns in result['dialogue'].values():
            for turn in turns:
                del turn['seconds']
    if (observed != reference or sha(candidate_path) != digest
            or frozen_digest(runtime.organism, runtime.state) != report['individual_sha256']):
        raise ValueError('The reloaded checkpoint did not reproduce the accepted result')
    return {'candidate_sha256': digest, 'accepted': report['accepted'], 'identical': True,
            'scores': compare(report['parent'], observed)}


def screen(args, protected):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime, tokenizer = load(args.checkpoint)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    gates = {}
    for index in args.layers:
        layer = runtime.decoder.base.model.layers[index]
        if isinstance(layer, SPORAResidualBlock):
            layer = layer.base
        if (not isinstance(layer.mlp, SPORAGatedFeedForward)
                or not isinstance(layer.mlp.activation, (SiLUActivation, torch.nn.SiLU))):
            raise ValueError(f'Layer {index} does not use the original SiLU gate')
        gates[index] = layer.mlp

    baseline = evaluate(runtime, tokenizer, args.ceval, 'parent')
    baseline_scores = compare(baseline, baseline)
    results = {'parent': baseline, 'parent_scores': baseline_scores, 'trials': []}
    accepted = []
    with ExitStack() as restore:
        for index, mlp in gates.items():
            original = mlp.activation
            restore.callback(setattr, mlp, 'activation', original)
            mlp.activation = SPORAGeluGate().eval()
            layers = [*accepted, index]
            candidate = evaluate(runtime, tokenizer, args.ceval, f'layers-{layers}')
            metrics = compare(baseline, candidate)
            passed = (metrics['memory_identical'] and all(metrics[key] >= baseline_scores[key]
                      for key in ('ceval', 'open_eos', 'short_cases', 'short_turns')))
            print(layers, metrics, 'passed', passed, flush=True)
            results['trials'].append({'layers': layers, 'accepted': passed, 'scores': metrics, 'evaluation': candidate})
            if passed:
                accepted.append(index)
            else:
                mlp.activation = original
        if accepted:
            candidate_path = runtime.save(args.output / 'candidate.pt')
            results['candidate_sha256'] = sha(candidate_path)
    if frozen_digest(runtime.organism, runtime.state) != state_digest:
        raise ValueError('Gate screen changed the individual')
    results.update(accepted=accepted, individual_sha256=state_digest, sources=protected)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--layers', type=int, nargs='+')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.layers and (len(set(args.layers)) != len(args.layers)
                        or any(not 18 <= index < 28 for index in args.layers)):
        parser.error('Choose distinct late layers at or after index 18')
    destination = args.output / ('reload.json' if args.verify else 'result.json')
    if destination.exists():
        raise FileExistsError(destination)
    protected = inputs(args.checkpoint, args.ceval)
    result = verify(args.output, args.checkpoint, args.ceval, protected) if args.verify else screen(args, protected)
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Gate screen changed a protected source')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
