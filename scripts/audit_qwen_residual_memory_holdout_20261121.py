"""Compare a SPORA residual on disjoint people and current/history questions."""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_awake_multiturn_layers import score_memory
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.pretrained_attention import SPORAGeluGate, SPORAResidualBlock
from transformers import AutoTokenizer


PEOPLE = ('周棠', '梁净', '沈玥', '林渡', '陈葵', '郭岚')


def memory_cases(organism, offset):
    state = fresh_controlled_state(organism)
    rows = []
    for i, person in enumerate(PEOPLE):
        key = 36 + i
        event = state.cursor
        values = ((2 * i + offset) % len(WORDS), (2 * i + offset + 3) % len(WORDS))
        for value in values:
            if state.cursor >= len(state.pages) * organism.config.time_count:
                state = grow(organism, state)
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.)
            if not decision['current_write']:
                raise ValueError('Controlled write was rejected')
        rows.extend((
            (f'按手头最新的备注，{person}平时爱喝哪一种？', key, 'current', event, values[1]),
            (f'请翻到改口前，{person}最常喝什么饮料？', key, 'past', event, values[0]),
        ))
    return [(question, WORDS[value], _memory_args(organism, state, {'key': key, 'intent': mode, 'event_time': event}), None)
            for question, key, mode, event, value in rows]


@contextmanager
def silu_control(decoder, layer_index):
    if layer_index not in decoder.compatible_block_indices:
        raise ValueError('The SiLU control requires a compatible layer')
    layer = decoder.base.model.layers[layer_index]
    if layer_index in decoder.layer_indices:
        layer = layer.layer
    if isinstance(layer, SPORAResidualBlock):
        layer = layer.base
    original = layer.mlp.activation
    if not isinstance(original, SPORAGeluGate):
        raise ValueError('The selected layer must already use the SPORA GELU gate')
    layer.mlp.activation = torch.nn.SiLU().eval()
    try:
        yield
    finally:
        layer.mlp.activation = original


def score_state_pair(decoder, tokenizer, original, changed):
    first = score_memory(decoder, tokenizer, original)
    second = score_memory(decoder, tokenizer, changed)
    summary = {'questions': len(first), 'answers_correct': sum(row['correct'] for row in first + second),
               'switches_correct': sum(a['correct'] and b['correct'] and a['expected'] not in b['answer']
                                       and b['expected'] not in a['answer']
                                       for a, b in zip(first, second, strict=True))}
    return {'first': first, 'second': second, 'summary': summary}


@torch.inference_mode()
def main(checkpoint, output, silu_layer=None):
    if output.exists():
        raise FileExistsError(output)
    digest = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, organism, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    original, changed = (memory_cases(organism, offset) for offset in (0, 2))
    residuals = [block for block in decoder.modules() if isinstance(block, SPORAResidualBlock)]
    if not residuals:
        raise ValueError('Expected a residual candidate')
    results = {'trained': score_state_pair(decoder, tokenizer, original, changed)}
    if silu_layer is not None:
        with silu_control(decoder, silu_layer):
            results['silu'] = score_state_pair(decoder, tokenizer, original, changed)
    else:
        for block in residuals:
            block.up.weight.zero_()
        results['disabled'] = score_state_pair(decoder, tokenizer, original, changed)
    for condition, result in results.items():
        print(condition, result['summary'], flush=True)
    if sha(checkpoint) != digest:
        raise ValueError('The source checkpoint changed during evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint_sha256': digest, 'results': results}
    if silu_layer is not None:
        report['silu_control_layer'] = silu_layer
    write(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--silu-layer', type=int, help='Restore one GELU gate to SiLU with residuals held fixed')
    args = parser.parse_args()
    main(args.checkpoint, args.output, args.silu_layer)
