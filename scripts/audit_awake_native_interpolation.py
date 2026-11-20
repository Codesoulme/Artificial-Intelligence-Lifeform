"""Compare interpolation between checkpoints of the same native decoder block."""

import argparse
import time
from pathlib import Path

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_memory
from audit_awake_native_block import INDEX, evaluate_with_value_switches
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from spora.structure_transfer import load_native_block
from train_awake_replacement_replay import load_layer_pair


FRACTIONS = (0., .25, .5, .75, 1.)
FRESH = ((11, '\u6587\u821f', 0, 5), (12, '\u82b3\u534e', 3, 1),
         (13, '\u5c0f\u82b8', 4, 6), (14, '\u9a86\u79be', 7, 2))


def paired_states(organism, facts, offset=2):
    _, first = history_cases(organism, facts)
    shifted = [(key, name, (earlier + offset) % 8, (current + offset) % 8)
               for key, name, earlier, current in facts]
    _, second = history_cases(organism, shifted)
    return first, second


def count_switches(first, second):
    return sum(a['correct'] and b['correct'] and a['expected'] not in b['answer']
               and b['expected'] not in a['answer'] for a, b in zip(first, second, strict=True))


def main(output, start, end, ceval):
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             start, end, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in paths}
    started = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    base = decoder.base
    checkpoint = dict(checkpoint_digest=protected[SOURCE], base_digest=protected[MODEL / 'model.safetensors'])
    blocks = []
    for path in (start, end):
        digest = read(path.parent / 'result.json')['candidate_sha256']
        blocks.append(load_native_block(path, base, INDEX, candidate_digest=digest, **checkpoint))
    original, distilled = blocks
    if original.max_length != distilled.max_length:
        raise ValueError('Expected checkpoints with the same position range')
    left = {name: tensor.detach().clone() for name, tensor in original.state_dict().items()}
    right = distilled.state_dict()
    del distilled, blocks
    base.model.layers[INDEX] = original
    questions = list(ceval_questions(ceval, tokenizer))
    memory, alternate = paired_states(organism, HISTORY)
    fresh, fresh_alternate = paired_states(organism, FRESH)
    results = {}
    for fraction in FRACTIONS:
        key = str(fraction)
        original.load_state_dict({name: torch.lerp(tensor, right[name], fraction) for name, tensor in left.items()})
        row = evaluate_with_value_switches(decoder, tokenizer, memory, alternate, questions, original.max_length)
        first = score_memory(decoder, tokenizer, fresh)
        second = score_memory(decoder, tokenizer, fresh_alternate)
        row['summary']['fresh_switches'] = count_switches(first, second)
        row['fresh_memory'], row['fresh_alternate'] = first, second
        results[key] = row
        print(key, row['summary'], flush=True)

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during evaluation')
    baseline = results['0.0']['summary']
    permitted = [fraction for fraction in FRACTIONS[1:] if all(
        results[str(fraction)]['summary'][metric] >= baseline[metric]
        for metric in ('memory_correct', 'memory_switches', 'fresh_switches', 'ceval', 'dialogue_correct')
    )]
    best = max(permitted, key=lambda fraction: (results[str(fraction)]['summary']['ceval'],
                                                results[str(fraction)]['summary']['dialogue_correct'], -fraction),
               default=0.)
    digest = None
    if best and any(results[str(best)]['summary'][metric] > baseline[metric]
                    for metric in ('ceval', 'dialogue_correct')):
        candidate = output / f'layer-{INDEX}.pt'
        metadata = dict(format='spora-qwen-layer-replacement-candidate-v1', layer=INDEX, width=original.width,
                        heads=original.block.heads, max_length=original.max_length,
                        base_weight_sha256=checkpoint['base_digest'], source_sha256=checkpoint['checkpoint_digest'])
        state = {name: torch.lerp(tensor, right[name], best).cpu() for name, tensor in left.items()}
        atomic_save(dict(metadata=metadata, state=state), candidate)
        digest = sha(candidate)
        base.model.layers[INDEX] = load_native_block(candidate, base, INDEX, candidate_digest=digest, **checkpoint)
        reloaded = evaluate_with_value_switches(decoder, tokenizer, memory, alternate, questions, original.max_length)
        reloaded['fresh_memory'] = score_memory(decoder, tokenizer, fresh)
        reloaded['fresh_alternate'] = score_memory(decoder, tokenizer, fresh_alternate)
        reloaded['summary']['fresh_switches'] = count_switches(reloaded['fresh_memory'], reloaded['fresh_alternate'])
        if reloaded != results[str(best)]:
            raise ValueError('Reloaded interpolation changed evaluation results')

    write(output / 'result.json', dict(results=results, selected_fraction=best if digest else None,
                                      candidate_sha256=digest, sources={str(p): h for p, h in protected.items()},
                                      seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--end', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    main(args.output, args.start, args.end, args.ceval)
