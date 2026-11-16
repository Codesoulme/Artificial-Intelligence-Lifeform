"""Train a shallow writer against the frozen language head on sampled tokens.

Oracle value boundaries and synthetic facts make this a research diagnostic.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_writer_layer_control import collect, create_cases, make_writer
from spora.language_training import atomic_save
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import value_positions

OUT = ROOT / 'runs/deepseek-spora-output-aware-writer-20261116'
SEED = 29741
STEPS = 512
NEGATIVES = 4096
BATCH = 16
GAIN = 32.


def sample_output_tokens(vocab_size, train_ids, held_ids, negatives, seed):
    if (len(set(train_ids)) != len(train_ids) or len(set(held_ids)) != len(held_ids)
            or set(train_ids) & set(held_ids) or any(not 0 <= i < vocab_size for i in train_ids + held_ids)):
        raise ValueError('Token splits must be unique, in range and disjoint')
    excluded = set(train_ids) | set(held_ids)
    unused = [i for i in range(vocab_size) if i not in excluded]
    if negatives > len(unused):
        raise ValueError('Not enough negative tokens')
    return list(train_ids) + random.Random(seed).sample(unused, negatives)


def norm_vectors(vectors, norm):
    return F.normalize(vectors, dim=-1) * norm


def hard_negative_ids(logits, existing, held, top_k):
    if logits.ndim != 2 or top_k < 1 or top_k > logits.shape[1]:
        raise ValueError('Expected a rank-two logit matrix and valid top_k')
    excluded = set(existing) | set(held)
    result = []
    for item in logits.topk(top_k, dim=-1).indices.flatten().cpu().tolist():
        if item not in excluded:
            excluded.add(item)
            result.append(item)
    return result


def main(output=OUT, hard_negatives=False):
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    pool = prior.stable_cjk_pool(tokenizer)
    random.Random(prior.SEED).shuffle(pool)
    train, held = pool[:prior.TRAIN_COUNT], pool[prior.TRAIN_COUNT:prior.TRAIN_COUNT + prior.HELD_COUNT]
    cases = create_cases(prior.TRAIN_NAMES, prior.HELD_NAMES, train, held)
    model = load_full_streamed().eval()
    features, targets = collect(model, tokenizer, cases)
    train_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in train]
    held_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in held]
    sampled_ids = sample_output_tokens(model.config.vocab_size, train_ids, held_ids, NEGATIVES, SEED)
    sampled_head = model.lm_head.weight.detach()[sampled_ids].float()
    train_norm = float(model.model.embed_tokens.weight[train_ids].float().norm(dim=-1).mean())
    prompt_hidden = {}
    for person in prior.TRAIN_NAMES + prior.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{person}喜欢喝什么饮料？', 'cuda')
        with torch.inference_mode():
            prompt_hidden[person] = model.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[0, -1].float().cpu().clone()
    indexes = [i for i, row in enumerate(cases) if row['name_split'] == 'train' and row['value_split'] == 'train']
    training = features['shallow_layer_1'][indexes].to('cuda')
    prompts = torch.stack([prompt_hidden[cases[i]['name']] for i in indexes]).to('cuda')
    value_index = {value: i for i, value in enumerate(train)}
    labels = torch.tensor([value_index[cases[i]['value']] for i in indexes], device='cuda')
    writer = make_writer(model.config.hidden_size, SEED, True).to('cuda')
    optimizer = torch.optim.AdamW(writer.parameters(), lr=prior.LR, weight_decay=1e-5, foreach=False)
    sampler = torch.Generator(device='cpu').manual_seed(SEED)
    trace = []
    for step in range(STEPS):
        batch = torch.randint(len(indexes), (BATCH,), generator=sampler).to('cuda')
        vector = norm_vectors(writer(training.index_select(0, batch)), train_norm)
        hidden = prompts.index_select(0, batch) + GAIN * vector
        if hard_negatives and step % 64 == 0:
            with torch.no_grad():
                full = model.lm_head(hidden.detach().to(model.lm_head.weight.dtype)).float()
                extra = hard_negative_ids(full, sampled_ids, held_ids, 8)
                if extra:
                    sampled_ids.extend(extra)
                    sampled_head = torch.cat((sampled_head, model.lm_head.weight.detach()[extra].float()))
        logits = F.linear(hidden, sampled_head)
        loss = F.cross_entropy(logits, labels.index_select(0, batch))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(writer.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 64 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient': float(gradient),
                          'output_classes': len(sampled_ids)})
            print(json.dumps(trace[-1]), flush=True)
    install_sequence_scaled(model, scale=GAIN)
    rows = []
    for person in prior.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{person}喜欢喝什么饮料？', 'cuda')
        for value in held[:8]:
            encoded, positions = value_positions(tokenizer, f'{person}喜欢喝{value}。', value)
            if len(positions) != 1:
                raise ValueError('Expected one token value')
            encoded = {k: v.to('cuda') for k, v in encoded.items()}
            token = int(encoded['input_ids'][0, positions[0]])
            with torch.inference_mode():
                fact = model.model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                vector = norm_vectors(writer(fact.hidden_states[1][0, positions[0]].float()), train_norm)
                eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float()
                with use_slot_sequence(torch.stack((vector, eos))):
                    generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=4,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
            rows.append({'person': person, 'value': value, 'token': token, 'generated_ids': tail,
                         'first_correct': bool(tail and tail[0] == token),
                         'exact': tail == [token, tokenizer.eos_token_id]})
        print(json.dumps({'completed_person': person, 'cases': len(rows)}, ensure_ascii=False), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Base weight changed')
    output.mkdir()
    path = output / 'writer.pt'
    atomic_save({'format': 'spora-output-aware-writer-v1', 'base_sha256': base.WEIGHT_SHA,
                 'state': {k: v.detach().cpu() for k, v in writer.state_dict().items()},
                 'train_norm': train_norm, 'seed': SEED}, path)
    report = {'base_sha256': base.WEIGHT_SHA, 'writer_sha256': base.sha(path), 'rows': rows, 'trace': trace,
              'summary': {'first_correct': sum(r['first_correct'] for r in rows),
                          'full_exact': sum(r['exact'] for r in rows), 'cases': len(rows)},
              'train_values': train, 'held_values': held, 'sampled_negative_count': NEGATIVES,
              'hard_negatives': hard_negatives, 'final_output_classes': len(sampled_ids),
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Frozen base, oracle boundaries and synthetic one-token values.',
                              'Training uses sampled output vocabulary; held values excluded from targets and negatives.',
                              'Only two held names and first eight held tokens are generated.',
                              'EOS is supplied by an oracle; no autonomous routing or persistence.',
                              'No replaced SPORA layers or open dialogue assessment.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--hard-negatives', action='store_true')
    args = parser.parse_args()
    main(args.output, args.hard_negatives)
