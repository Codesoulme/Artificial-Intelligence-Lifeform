"""Replay a bounded bank of persistent tensors through one native attention reader."""

import argparse
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


def attend_with_slots(query, key, value, slots, mask, scale):
    """Let persistent and current keys compete in the same softmax."""
    saved_key, saved_value = slots
    count = saved_key.shape[-2]
    key = torch.cat((saved_key.to(key), key), dim=-2)
    value = torch.cat((saved_value.to(value), value), dim=-2)
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    if mask is not None:
        current = mask[..., -1:, :]
        padding = (0 if current.dtype.is_floating_point else True)
        mask = torch.nn.functional.pad(current, (count, 0), value=padding)
    return torch.nn.functional.scaled_dot_product_attention(
        query[..., -1:, :], key, value, attn_mask=mask, scale=scale,
    )


def read_with_slots(decoder, attention, ids, slots):
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
        mixed = attend_with_slots(query, key, value, slots, attention_mask, attention.scaling)
        result = normal.clone()
        result[:, -1:, :] = attention.out(mixed.transpose(1, 2).flatten(start_dim=-2)).to(normal)
        return result, weights

    input_ids = torch.tensor([ids], device=decoder.base.device)
    # The reserved positions give previously written keys a fixed, causal position before the query.
    positions = torch.arange(SLOT_BUDGET, SLOT_BUDGET + len(ids), device=input_ids.device)[None]
    with patch.object(attention, 'forward', augmented):
        logits = decoder(input_ids=input_ids, position_ids=positions, use_cache=False, logits_to_keep=1).logits
    return logits[0, -1].float().log_softmax(dim=-1)


def record_ids(tokenizer, topic, left, right, mapping):
    text = f'{topic}的字段编号已经确认。{left}的编号是{1 + mapping}，{right}的编号是{2 - mapping}。'
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                              tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(formatted, add_special_tokens=False)
    if not 1 <= len(ids) <= SLOT_BUDGET:
        raise ValueError('The evidence exceeds the fixed memory slot budget')
    return ids


def write_banks(decoder, tokenizer, attentions):
    return {
        layer: {
            topic: [capture_record(decoder, attention, record_ids(tokenizer, topic, left, right, mapping))
                    for mapping in (0, 1)]
            for topic, left, right in query_audit.FIELDS
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
                empty = read_with_slots(decoder, attention, ids, None)
                conditions = []
                for mapping, slots in enumerate(banks[layer][topic]):
                    expected = mapping ^ field
                    scores = read_with_slots(decoder, attention, ids, slots)
                    conditions.append({'mapping': mapping, 'target': expected + 1,
                                       'empty': query_audit.readout(empty, label_ids, expected),
                                       'stored': query_audit.readout(scores, label_ids, expected),
                                       'visible': baseline[mapping]})
                rows.append({'topic': topic, 'field': name, 'layer': layer, 'reads': conditions})
    return rows, label_ids


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
                'individual_sha256': individual, 'budget': SLOT_BUDGET}
    with using(candidates, 'recurrent'):
        if args.output is None:
            payload = {**identity, 'banks': write_banks(decoder, tokenizer, attentions)}
        else:
            serialized = sha(args.artifact)
            payload = torch.load(args.artifact, map_location='cpu', weights_only=True)
            if any(payload[key] != value for key, value in identity.items()):
                raise ValueError('The written memory used a different model, script, or individual')
            rows, label_ids = measure(decoder, tokenizer, attentions, payload['banks'])
            if sha(args.artifact) != serialized:
                raise RuntimeError('The persistent slots changed during evaluation')
    if individual != frozen_digest(runtime.organism, runtime.state) or any(sha(p) != h for p, h in protected.items()):
        raise RuntimeError('A source or the individual changed during evaluation')
    if args.output is None:
        atomic_save(payload, args.artifact)
    else:
        store_json(args.output, {**identity, 'slots_sha256': serialized, 'rows': rows, 'label_ids': label_ids,
                                 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, help='Read a previously written bank and score the answers')
    args = parser.parse_args()
    if (args.output or args.artifact).exists():
        parser.error('Choose a new output path')
    if args.output is not None and not args.artifact.is_file():
        parser.error('The previously written slot bank is missing')
    main(args)
