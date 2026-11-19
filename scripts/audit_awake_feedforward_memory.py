"""Measure a pretrained feed-forward transplant with paired native memory states."""

import argparse
from pathlib import Path

from audit_awake_language_layers import load_layer_pair
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.integrated_open_base import SPORAGatedFeedForward
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputAdapter


ORDINARY = ('只回复两个字：你好。', '五张桌子，每桌三把椅子，一共有几把椅子？', '为什么冬天的金属把手摸起来比木头凉？')
MEMORY = ('记录里这个人最喜欢喝什么饮料？', '这个人的饮料偏好是什么？')
HISTORY = ((1, '小林', 2, 5), (2, '阿宁', 3, 4), (3, '小陈', 1, 7), (4, '小杨', 4, 0))


def history_cases(organism):
    state = fresh_controlled_state(organism)
    for key, _, earlier, current in HISTORY:
        for value in (earlier, current):
            state, event = observe_sourced(organism, state, key, value, 'direct', 1.)
            if not event['current_write']:
                raise ValueError('Controlled history write rejected')
    rows = []
    for index, (key, person, earlier, current) in enumerate(HISTORY):
        for intent, question, expected in (
            ('current', f'{person}现在最喜欢喝什么饮料？', current),
            ('past', f'{person}最早那次偏爱喝什么？', earlier),
        ):
            args = _memory_args(organism, state, dict(key=key, intent=intent, event_time=2 * index))
            device = next(organism.parameters()).device
            replay_time = 2 * index if intent == 'past' else 0
            readout = query(organism, state, torch.tensor([key], device=device),
                            torch.tensor([int(intent == 'past')], device=device),
                            torch.tensor([replay_time], device=device))
            rows.append((question, WORDS[expected], args, readout))
    return state, rows


def load_candidate(path, base_hash, decoder):
    report = read(path.parent / 'result.json')
    digest = sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (report['adapter_sha256'] != digest or payload['format'] != 'spora-memory-output-adapter-v1'
            or payload['base_sha256'] != base_hash or payload['config'] != decoder.output_adapter.config_dict()):
        raise ValueError('Output candidate provenance or layout mismatch')
    adapter = MemoryOutputAdapter(**payload['config'])
    adapter.load_state_dict(payload['state'], strict=True)
    return adapter.to(decoder.output_adapter.value_to_rank.weight.device).requires_grad_(False).eval(), digest


@torch.inference_mode()
def main(output, layer_index, fractions, candidate_path):
    source_hash, base_hash = sha(SOURCE), sha(MODEL/'model.safetensors')
    decoder, tokenizer, _, _, organism, individual = load_layer_pair()
    if layer_index not in decoder.layer_indices:
        raise ValueError('Expected one of the checkpoint memory layers')
    wrapper = decoder.base.model.layers[layer_index]
    original = frozen_digest(organism, individual)
    states = []
    for value in (1, 6):
        state, decision = observe_sourced(organism, fresh_controlled_state(organism), 0, value, 'direct', 1.)
        if not decision['current_write']:
            raise ValueError('Controlled native write rejected')
        states.append(state)
    state_digests = tuple(frozen_digest(organism, state) for state in states)
    args_by_value = [_memory_args(organism, state, dict(key=0, intent='current')) for state in states]
    historical_state, historical = history_cases(organism)
    historical_digest = frozen_digest(organism, historical_state)

    def prompt(text):
        rendered = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                                                 add_generation_prompt=True) + '</think>\n\n'
        return tokenizer.encode(rendered, add_special_tokens=False, return_tensors='pt').to('cuda')

    def run(text, memory_args, output_logits=None):
        ids = prompt(text)
        mask = torch.ones_like(ids)
        forward_args = dict(memory_args, use_cache=False)
        generation_args = dict(memory_args, use_cache=True)
        logits = decoder(input_ids=ids, attention_mask=mask, **forward_args).logits[0, -1].float().cpu()
        generation = decoder.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                       top_p=None, temperature=None, max_new_tokens=32, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id, memory_output_logits=output_logits,
                                       **generation_args)
        tokens = generation[0, ids.shape[1]:].tolist()
        return logits, dict(answer=tokenizer.decode(tokens, skip_special_tokens=True).strip(), tokens=tokens)

    ordinary_before = [run(q, {}) for q in ORDINARY]
    memory_before = [[run(q, args) for args in args_by_value] for q in MEMORY]
    history_before = [run(question, args)[1] for question, _, args, _ in historical] if fractions else []
    history_adapted = [run(question, args, readout)[1] for question, _, args, readout in historical] if fractions else []
    original_adapter = decoder.output_adapter
    candidate_digest = None
    if candidate_path is not None:
        candidate, candidate_digest = load_candidate(candidate_path, base_hash, decoder)
        decoder.output_adapter = candidate
        history_candidate = [run(question, args, readout)[1] for question, _, args, readout in historical]
        decoder.output_adapter = original_adapter
    original_mlp = wrapper.layer.mlp
    wrapper.layer.mlp = SPORAGatedFeedForward(original_mlp).eval()
    fused_memory = [[run(q, args) for args in args_by_value] for q in MEMORY]
    wrapper.layer.mlp = original_mlp
    decoder.use_feedforward_memory(layer_index)
    ordinary_after = [run(q, {}) for q in ORDINARY]
    memory_after = [[run(q, args) for args in args_by_value] for q in MEMORY]
    blends = {}
    history = {}
    for fraction in fractions:
        decoder.use_feedforward_memory(layer_index, fraction=fraction)
        rows = [[run(question, args)[1] for args in args_by_value] for question in MEMORY]
        blends[fraction] = rows
        history[fraction] = [dict(question=question, expected=expected, matrix_value=int(readout[0].argmax()),
                                  before=before, after=run(question, args)[1], before_adapted=adapted,
                                  after_adapted=run(question, args, readout)[1])
                             for (question, expected, args, readout), before, adapted
                             in zip(historical, history_before, history_adapted, strict=True)]
        if candidate_path is not None:
            decoder.output_adapter = candidate
            for row, (question, _, args, readout), before in zip(history[fraction], historical, history_candidate, strict=True):
                row['before_candidate'] = before
                row['after_candidate'] = run(question, args, readout)[1]
            decoder.output_adapter = original_adapter
            print('Candidate history', fraction, [(row['expected'], row['before_candidate']['answer'],
                                                    row['after_candidate']['answer']) for row in history[fraction]], flush=True)
        print('Fraction', fraction, [[row['answer'] for row in pair] for pair in rows], flush=True)
        print('History', fraction, [(row['expected'], row['before']['answer'], row['after']['answer'])
                                    for row in history[fraction]], flush=True)
        print('Adapted history', fraction, [(row['expected'], row['before_adapted']['answer'], row['after_adapted']['answer'])
                                            for row in history[fraction]], flush=True)
    ordinary = []
    for question, (logits_a, a), (logits_b, b) in zip(ORDINARY, ordinary_before, ordinary_after, strict=True):
        ordinary.append(dict(question=question, before=a, after=b, same_tokens=a['tokens'] == b['tokens'],
                             max_logit_difference=(logits_a - logits_b).abs().max().item()))
    memory = []
    for question, before, fused, after in zip(MEMORY, memory_before, fused_memory, memory_after, strict=True):
        memory.append(dict(question=question, before=[row for _, row in before], fused=[row for _, row in fused],
                           after=[row for _, row in after],
                           before_pair_logit_difference=(before[0][0] - before[1][0]).abs().max().item(),
                           after_pair_logit_difference=(after[0][0] - after[1][0]).abs().max().item()))
    if (sha(SOURCE) != source_hash or sha(MODEL/'model.safetensors') != base_hash
            or (candidate_path is not None and sha(candidate_path) != candidate_digest)
            or original != frozen_digest(organism, individual)
            or historical_digest != frozen_digest(organism, historical_state)
            or state_digests != tuple(frozen_digest(organism, state) for state in states)):
        raise ValueError('A protected checkpoint or state changed during the audit')
    write(output, dict(layer=layer_index, reading_layers=decoder.layer_indices, ordinary=ordinary, memory=memory,
                       blends=blends, history=history,
                       source_sha256=source_hash, base_sha256=base_hash, candidate_adapter_sha256=candidate_digest))
    print('Ordinary', [row['same_tokens'] for row in ordinary], 'Memory pairs', memory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--fraction', type=float, action='append', default=[])
    parser.add_argument('--candidate-adapter', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0. < fraction < 1. for fraction in args.fraction):
        parser.error('Each optional fraction must be between 0 and 1')
    if args.candidate_adapter is not None and not args.fraction:
        parser.error('--candidate-adapter requires a migration fraction for the paired audit')
    main(args.output, args.layer, args.fraction, args.candidate_adapter)
