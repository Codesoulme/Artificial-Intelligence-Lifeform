"""Replay a bounded bank of persistent tensors through one native attention reader."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from scripts import audit_spora_recurrent_query_binding as query_audit
from scripts.audit_spora_recurrent_attention import last_logits
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, sha, torch, using,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_recurrent_decode import attention_block, load_recurrent_replacements


LAYERS = (20, 23)
SLOT_BUDGET = 40
FORMAT = 'spora-persistent-attention-slots-v1'
LOGIT_BIASES = (0, 2, 4)
BASE_CASES = tuple((*fields, ('1', '2')) for fields in query_audit.FIELDS)
FRESH_CASES = (
    ('摊位排班', '摊位', '班次', ('3', '7')),
    ('仓单核对', '柜号', '批次', ('4', '9')),
)


def capture_record(decoder, attention, ids):
    captured = None
    project = attention.project

    def save(*args, **kwargs):
        nonlocal captured
        query, key, value = project(*args, **kwargs)
        captured = (key.detach().cpu(), value.detach().cpu())
        return query, key, value

    with patch.object(attention, 'project', save):
        last_logits(decoder, ids)
    if captured is None or captured[0].shape[-2] != len(ids):
        raise RuntimeError('The native reader did not capture the requested record')
    return captured


def attend_with_slots(query, key, value, slots, mask, scale, *, bias=0, capture_mass=None):
    """Let persistent and current keys compete in the same softmax."""
    saved_key, saved_value = slots
    count = saved_key.shape[-2]
    key = torch.cat((saved_key.to(key), key), dim=-2)
    value = torch.cat((saved_value.to(value), value), dim=-2)
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    query = query[..., -1:, :]
    if mask is not None:
        current = mask[..., -1:, :key.shape[-2] - count]
        padding = (0 if current.dtype.is_floating_point else True)
        mask = torch.nn.functional.pad(current, (count, 0), value=padding)
    if bias:
        if mask is None:
            mask = query.new_zeros((1, 1, 1, key.shape[-2]))
        elif mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(~mask, -torch.inf)
        else:
            mask = mask.to(query.dtype)
        mask[..., :count] += bias
    if capture_mass is not None:
        logits = (query.float() @ key.float().transpose(-2, -1)) * scale
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else logits + mask
        capture_mass(logits.softmax(-1)[0, :, 0, :count].sum(-1).tolist())
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale,
    )


def augment_attention(attention, slots, bias, capture_mass):
    forward = attention.forward

    def augmented(hidden_states, position_embeddings, attention_mask, past_key_values=None,
                  cache_position=None, position_ids=None, **kwargs):
        if past_key_values is not None:
            raise ValueError('The pilot measures the first prediction from fresh input')
        normal, weights = forward(hidden_states, position_embeddings, attention_mask,
                                  past_key_values=past_key_values, cache_position=cache_position,
                                  position_ids=position_ids, **kwargs)
        if slots is None:
            return normal, weights
        query, key, value = attention.project(
            hidden_states.to(attention.qkv.weight.dtype), position_embeddings, position_ids,
        )
        mixed = attend_with_slots(
            query, key, value, slots, attention_mask, attention.scaling, bias=bias, capture_mass=capture_mass,
        )
        result = normal.clone()
        result[:, -1:, :] = attention.out(mixed.transpose(1, 2).flatten(start_dim=-2)).to(normal)
        return result, weights

    return augmented


def read_attention(decoder, ids, sites, *, bias=0, mass=None):
    input_ids = torch.tensor([ids], device=decoder.base.device)
    # The reserved positions give previously written keys a fixed, causal position before the query.
    positions = torch.arange(SLOT_BUDGET, SLOT_BUDGET + len(ids), device=input_ids.device)[None]
    with ExitStack() as hooks:
        for attention, slots in sites:
            capture = None if mass is None else mass[attention.layer_idx].extend
            hooks.enter_context(patch.object(attention, 'forward', augment_attention(attention, slots, bias, capture)))
        logits = decoder(input_ids=input_ids, position_ids=positions, use_cache=False, logits_to_keep=1).logits
    return logits[0, -1].float().log_softmax(dim=-1)


def record_text(topic, left, right, mapping, labels):
    return f'{topic}的字段编号已经确认。{left}的编号是{labels[mapping]}，{right}的编号是{labels[1 - mapping]}。'


def record_ids(tokenizer, topic, left, right, mapping, labels):
    text = record_text(topic, left, right, mapping, labels)
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                              tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(formatted, add_special_tokens=False)
    if not 1 <= len(ids) <= SLOT_BUDGET:
        raise ValueError('The evidence exceeds the fixed memory slot budget')
    return ids


def write_banks(decoder, tokenizer, attentions, cases):
    return {
        layer: {
            topic: [capture_record(decoder, attention, record_ids(tokenizer, topic, left, right, mapping, labels))
                    for mapping in (0, 1)]
            for topic, left, right, labels in cases
        }
        for layer, attention in attentions.items()
    }


def measure(decoder, tokenizer, attentions, banks):
    labels = [tokenizer.encode(value, add_special_tokens=False) for value in ('1', '2')]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError('Expected distinct single-token labels')
    label_ids = [ids[0] for ids in labels]
    suffix = tokenizer.encode('编号：', add_special_tokens=False)
    visible = dict(query_audit.questions(tokenizer, []))
    rows = []
    for topic, left, right in query_audit.FIELDS:
        for field, name in enumerate((left, right)):
            request = {'role': 'user', 'content': f'回到{topic}的登记：{name}对应哪个编号？请只填编号。'}
            ids = encode(tokenizer, [request]) + suffix
            if any(token in ids for token in label_ids):
                raise ValueError('A target label appeared in the fresh request')
            baseline = [
                query_audit.readout(last_logits(decoder, visible[topic][mapping, field]), label_ids, mapping ^ field)
                for mapping in (0, 1)
            ]
            for layer, attention in attentions.items():
                empty = read_attention(decoder, ids, ((attention, None),))
                conditions = []
                for mapping, slots in enumerate(banks[layer][topic]):
                    expected = mapping ^ field
                    scores = read_attention(decoder, ids, ((attention, slots),))
                    conditions.append({'mapping': mapping, 'target': expected + 1,
                                       'empty': query_audit.readout(empty, label_ids, expected),
                                       'stored': query_audit.readout(scores, label_ids, expected),
                                       'visible': baseline[mapping]})
                rows.append({'topic': topic, 'field': name, 'layer': layer, 'reads': conditions})
    return rows, label_ids


def sweep(decoder, tokenizer, attentions, banks):
    cases = BASE_CASES + FRESH_CASES
    count = min(key.shape[-2] for bank in banks.values() for versions in bank.values() for key, _ in versions)
    suffix = tokenizer.encode('编号：', add_special_tokens=False)
    rows = []
    for index, (topic, left, right, labels) in enumerate(cases):
        foreign_topic = cases[(index + 1) % len(cases)][0]
        label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(tokens) != 1 for tokens in label_tokens) or label_tokens[0] == label_tokens[1]:
            raise ValueError('Expected two distinct single-token labels')
        label_ids = [tokens[0] for tokens in label_tokens]
        fresh = index >= len(BASE_CASES)
        for field, name in enumerate((left, right)):
            question = (f'我想核对{topic}留下的{name}编号。能只报数字吗？' if fresh
                        else f'回到{topic}的登记：{name}对应哪个编号？请只填编号。')
            request = {'role': 'user', 'content': question}
            ids = encode(tokenizer, [request]) + suffix
            if any(token in ids for token in label_ids):
                raise ValueError('An answer label appeared in the fresh request')
            empty = read_attention(decoder, ids, ())
            visibility = []
            for mapping in (0, 1):
                history = [{'role': 'user', 'content': record_text(topic, left, right, mapping, labels)},
                           {'role': 'assistant', 'content': '好，我已登记。'}, request]
                visibility.append(last_logits(decoder, encode(tokenizer, history) + suffix))
            for layers in ((23,), (20, 23)):
                row = {'topic': topic, 'field': name, 'fresh': fresh, 'layers': layers,
                       'foreign_topic': foreign_topic, 'label_ids': label_ids, 'conditions': []}
                for bias in LOGIT_BIASES:
                    for loaded_topic, compatible in ((topic, True), (foreign_topic, False)):
                        for mapping in (0, 1):
                            sites = [
                                (attentions[layer], tuple(part[..., -count:, :] for part in banks[layer][loaded_topic][mapping]))
                                for layer in layers
                            ]
                            mass = {layer: [] for layer in layers}
                            scores = read_attention(decoder, ids, sites, bias=bias, mass=mass)
                            expected = mapping ^ field
                            row['conditions'].append({
                                'bias': bias, 'compatible': compatible, 'mapping': mapping,
                                'target': label_ids[expected], 'mass': mass,
                                'empty': query_audit.readout(empty, label_ids, expected),
                                'visible': query_audit.readout(visibility[mapping], label_ids, expected),
                                'stored': query_audit.readout(scores, label_ids, expected),
                            })
                rows.append(row)
        print(topic, 'memory slots per layer', count, flush=True)
    return rows, count


@torch.inference_mode()
def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    protected = {path: sha(path) for path in (
        args.checkpoint, Path(__file__), Path(query_audit.__file__), MODEL / 'model.safetensors', MODEL / 'tokenizer.json',
        MODEL / 'tokenizer_config.json',
        *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')),
    )}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    native_rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, native_rotary, query_audit.POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if tuple(candidates) != (19, 22):
        raise ValueError('Expected the established recurrent layout')
    attentions = {layer: attention_block(decoder, layer).self_attn for layer in LAYERS}
    identity = {'format': FORMAT, 'sources': {str(p): h for p, h in protected.items()},
                'individual_sha256': individual, 'budget': SLOT_BUDGET, 'sweep': args.sweep}
    with using(candidates, 'recurrent'):
        if args.output is None:
            cases = BASE_CASES + FRESH_CASES if args.sweep else BASE_CASES
            payload = {**identity, 'banks': write_banks(decoder, tokenizer, attentions, cases)}
        else:
            serialized = sha(args.artifact)
            payload = torch.load(args.artifact, map_location='cpu', weights_only=True)
            if any(payload[key] != value for key, value in identity.items()):
                raise ValueError('The written memory used a different model, script, or individual')
            if args.sweep:
                rows, count = sweep(decoder, tokenizer, attentions, payload['banks'])
                details = {'slots_per_layer': count, 'biases': LOGIT_BIASES}
            else:
                rows, label_ids = measure(decoder, tokenizer, attentions, payload['banks'])
                details = {'label_ids': label_ids}
            if sha(args.artifact) != serialized:
                raise RuntimeError('The persistent slots changed during evaluation')
    if individual != frozen_digest(runtime.organism, runtime.state) or any(sha(p) != h for p, h in protected.items()):
        raise RuntimeError('A source or the individual changed during evaluation')
    if args.output is None:
        atomic_save(payload, args.artifact)
    else:
        store_json(args.output, {**identity, **details, 'slots_sha256': serialized, 'rows': rows,
                                 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, help='Read a previously written bank and score the answers')
    parser.add_argument('--sweep', action='store_true', help='Audit memory attention mass across matched conditions')
    args = parser.parse_args()
    if (args.output or args.artifact).exists():
        parser.error('Choose a new output path')
    if args.output is not None and not args.artifact.is_file():
        parser.error('The previously written slot bank is missing')
    main(args)
