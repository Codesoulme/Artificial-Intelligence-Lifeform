"""Reload the migrated recurrent layer and evaluate ordinary and persistent replies."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_recurrent_attention import (
    MODEL, AttentionReplacement, AutoTokenizer, DynamicCache, RecurrentAttention, RecurrentDecodeLayer,
    attention_block, frozen_digest, load_full_streamed, load_integrated_runtime, replace, sha, torch,
)
from scripts.audit_spora_chat_backbones import reply
from scripts.audit_spora_dialogue_working_state import score_case
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from scripts.train_spora_dialogue_composition import settings_for


@torch.inference_mode()
def chat_rows(decoder, tokenizer, cases):
    settings = settings_for(decoder, tokenizer, 48)
    result = []
    for row in cases:
        case = row['case']
        generated = reply(decoder, tokenizer, row['history'], case['question'], **settings)
        generated.pop('seconds')
        result.append({'case': case['kind'], 'result': generated, **score_case(case, generated)})
    return result


@torch.inference_mode()
def cache_probe(decoder, ids, layer):
    device = decoder.base.device
    inputs = torch.tensor([ids], device=device)
    full = decoder(input_ids=inputs, use_cache=False, logits_to_keep=4).logits.float()
    cache = DynamicCache(config=decoder.base.config)
    rows = []
    for start, end in ((0, len(ids) - 4), *((index, index + 1) for index in range(len(ids) - 4, len(ids)))):
        output = decoder(input_ids=inputs[:, start:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
        if start >= len(ids) - 4:
            rows.append(output.logits.float())
    streamed = torch.cat(rows, dim=1)
    recurrent = cache.layers[layer]
    if not isinstance(recurrent, RecurrentDecodeLayer) or recurrent.state.length != len(ids):
        raise ValueError('The recurrent layer did not handle the complete cached sequence')
    torch.testing.assert_close(streamed, full, rtol=.025, atol=.08)
    if not torch.equal(streamed.argmax(dim=-1), full.argmax(dim=-1)):
        raise ValueError('The recurrent cached predictions changed their top tokens')
    state = recurrent.state
    storage = sum(tensor.numel() * tensor.element_size()
                  for tensor in (state.key, state.value, state.matrix, state.normalizer))
    return {'tokens': len(ids), 'recent_tokens': state.key.shape[-2], 'layer_bytes': storage,
            'max_logit_difference': float((streamed - full).abs().max()), 'matching_top_tokens': 4}


def main(args):
    destination = args.migration / 'reload.json'
    if destination.exists():
        raise FileExistsError(destination)
    candidate_path = args.migration / 'attention.pt'
    reference_path = args.migration / 'comparison.json'
    protected = {path: sha(path) for path in (args.checkpoint, args.revision, candidate_path, reference_path)}
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if (reference['candidate_sha256'] != protected[candidate_path]
            or reference['sources'][str(args.checkpoint)] != protected[args.checkpoint]
            or reference['sources'][str(args.revision)] != protected[args.revision]):
        raise ValueError('Recurrent candidate differs from the audited trial')
    saved = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if (saved['format'] != 'spora-recurrent-attention-v1' or saved['checkpoint_sha256'] != protected[args.checkpoint]
            or saved['layer'] != reference['layer'] or saved['block'] != reference['block']):
        raise ValueError('Recurrent candidate metadata changed')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    digest = frozen_digest(runtime.organism, runtime.state)
    if reference['individual_sha256'] != digest:
        raise ValueError('Individual differs from the audited trial')
    index = saved['layer']
    block = attention_block(decoder, index)
    source = block.self_attn
    core = RecurrentAttention(source.sizes[0] // source.head_dim, source.head_dim, block=saved['block'])
    core.load_state_dict(saved['core'], strict=True)
    core.to(decoder.base.device).eval().requires_grad_(False)
    replacement = AttentionReplacement(source, core)
    cases = json.loads(args.revision.read_text(encoding='utf-8'))['oracle']
    memories = records(runtime.organism, HELD)
    base_chat = chat_rows(decoder, tokenizer, cases)
    base_memory = evaluate(decoder, tokenizer, runtime.organism, memories)
    print('Parent', sum(row['complete'] for row in base_chat), base_memory['summary'], flush=True)
    long_text = '。'.join(row['case']['first'][0] + row['case']['update'][0] for row in cases)
    tokens = tokenizer.encode(long_text, add_special_tokens=False)
    with replace(block, replacement):
        cache = [cache_probe(decoder, tokens[:size], index) for size in (94, 190, 382)]
        print('Cache', cache, flush=True)
        migrated_chat = chat_rows(decoder, tokenizer, cases)
        migrated_memory = evaluate(decoder, tokenizer, runtime.organism, memories)
    if (frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A protected source or individual changed')
    result = {'cache': cache, 'parent': base_chat, 'candidate': migrated_chat,
              'memory_parent': base_memory, 'memory_candidate': migrated_memory,
              'memory_identical': base_memory == migrated_memory, 'individual_sha256': digest,
              'candidate_sha256': protected[candidate_path], 'peak_bytes': torch.cuda.max_memory_allocated()}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Candidate', sum(row['complete'] for row in migrated_chat), migrated_memory['summary'],
          'identical memories', result['memory_identical'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('revision', type=Path)
    parser.add_argument('migration', type=Path)
    main(parser.parse_args())
