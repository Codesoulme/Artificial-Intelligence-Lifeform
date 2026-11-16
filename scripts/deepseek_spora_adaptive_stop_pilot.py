"""Train a continuous read head to output a single-token fact and stop itself.

The caller supplies the relevant slot and value boundary; EOS is supervised
in training but no stop vector or step count is supplied during generation.
"""
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
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, hard_negative_ids, norm_vectors, sample_output_tokens
from scripts.deepseek_spora_writer_layer_control import collect
from spora.language_training import atomic_save
from spora.research_adaptive_semantic_head import AdaptiveSemanticHead, use_adaptive_slot
from spora.research_semantic_slot import SequenceSemanticSlotLMHead, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

SOURCE = ROOT / 'runs/deepseek-spora-output-aware-broad-20261116'
OUT = ROOT / 'runs/deepseek-spora-adaptive-stop-pilot-20261116'
SEED, STEPS, TRAIN_COUNT, BATCH = 28491, 1024, 256, 16


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    source = SOURCE / 'writer.pt'
    source_sha = base.sha(source)
    if source_report['base_sha256'] != base.WEIGHT_SHA or source_sha != source_report['writer_sha256']:
        raise ValueError('Writer source mismatch')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    train_values, held_values = source_report['train_values'][:TRAIN_COUNT], source_report['held_values']
    if len(train_values) != TRAIN_COUNT or set(train_values) & set(held_values):
        raise ValueError('Invalid train and held values')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    train_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in train_values]
    held_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in held_values]
    eos_id = tokenizer.eos_token_id
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    layer_hashes = {str(i): base.sha(path) for i, path in paths.items()}
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index, path in paths.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(saved['state'], strict=True)
    writer.requires_grad_(False)
    cases = [{'name': person, 'value': value} for person in prior.TRAIN_NAMES for value in train_values]
    features, _ = collect(model, tokenizer, cases)
    with torch.inference_mode():
        slots = norm_vectors(writer(features['shallow_layer_1'].to('cuda')), saved['train_norm']).cpu().clone()
    prompts = {}
    for name in prior.TRAIN_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{name}喜欢喝什么饮料？', 'cuda')
        with torch.inference_mode():
            initial = model.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[0, -1].float().cpu().clone()
        prompts[name] = (prompt['input_ids'][0].detach().cpu(), initial)
    hidden_pairs = []
    for start_idx in range(0, len(cases), 8):
        items = cases[start_idx:start_idx + 8]
        sequences = [torch.cat((prompts[item['name']][0], torch.tensor([train_ids[train_values.index(item['value'])]])))
                     for item in items]
        max_length = max(seq.numel() for seq in sequences)
        ids = torch.full((len(items), max_length), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :seq.numel()] = seq.to('cuda')
            mask[i, :seq.numel()] = 1
        positions = torch.tensor([seq.numel() - 1 for seq in sequences], device='cuda')
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 return_dict=True).last_hidden_state
            end = hidden[torch.arange(len(items), device='cuda'), positions].float().cpu()
        hidden_pairs.extend(torch.stack((prompts[item['name']][1], end[i])) for i, item in enumerate(items))
    hidden_pairs = torch.stack(hidden_pairs).to('cuda')
    slots = slots.to('cuda')
    labels = torch.tensor([[train_ids.index(tokenizer.encode(item['value'], add_special_tokens=False)[0]), TRAIN_COUNT]
                           for item in cases], dtype=torch.long, device='cuda')
    sampled_ids = sample_output_tokens(model.config.vocab_size, train_ids + [eos_id], held_ids, 4096, SEED)
    sampled_head = model.lm_head.weight.detach()[sampled_ids].float()
    adaptive = AdaptiveSemanticHead(model.lm_head, rank=32, gain=GAIN).to('cuda').train()
    params = [p for name, p in adaptive.named_parameters() if not name.startswith('base.')]
    optimizer = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-5, foreach=False)
    sampler = torch.Generator(device='cpu').manual_seed(SEED)
    trace = []
    for step in range(STEPS):
        indexes = torch.randint(len(cases), (BATCH,), generator=sampler).to('cuda')
        selected_hidden = hidden_pairs.index_select(0, indexes)
        selected_slots = slots.index_select(0, indexes)
        adjusted = adaptive.adjusted(selected_hidden, selected_slots)
        if step % 64 == 0:
            with torch.no_grad():
                full_logits = adaptive.base(adjusted.detach().to(adaptive.base.weight.dtype)).float().reshape(-1, model.config.vocab_size)
                extra = hard_negative_ids(full_logits, sampled_ids, held_ids, 8)
                if extra:
                    sampled_ids.extend(extra)
                    sampled_head = torch.cat((sampled_head, adaptive.base.weight.detach()[extra].float()))
        logits = F.linear(adjusted, sampled_head)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.index_select(0, indexes).reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 128 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'gradient_norm': float(gradient), 'classes': len(sampled_ids)})
            print(json.dumps(trace[-1]), flush=True)
    adaptive.eval()
    base_head = adaptive.base
    one_shot = SequenceSemanticSlotLMHead(base_head, scale=GAIN)
    rows = []
    for name in prior.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{name}喜欢喝什么饮料？', 'cuda')
        for value in held_values[:8]:
            encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
            encoded = {k: v.to('cuda') for k, v in encoded.items()}
            target = int(encoded['input_ids'][0, positions[0]])
            with torch.inference_mode():
                hidden = model.model(**encoded, output_hidden_states=True, use_cache=False,
                                     return_dict=True).hidden_states[1][0, positions].float()
                slot = norm_vectors(writer(hidden), saved['train_norm'])[0]
                outputs = {}
                for mode, head in (('one_shot', one_shot), ('adaptive', adaptive)):
                    model.lm_head = head
                    manager = use_slot_sequence(slot.unsqueeze(0)) if mode == 'one_shot' else use_adaptive_slot(slot)
                    with manager:
                        generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=16,
                                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_id)
                    tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                    outputs[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                                     'first_correct': bool(tail and tail[0] == target),
                                     'ended': bool(tail and tail[-1] == eos_id),
                                     'exact': tail == [target, eos_id]}
                rows.append({'name': name, 'value': value, 'target': target, 'outputs': outputs})
        print(json.dumps({'evaluated_name': name}, ensure_ascii=True), flush=True)
    ordinary = []
    for question in ('你好。', '一加一等于几？', '请给我一句休息建议。'):
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        outcomes = {}
        with torch.inference_mode():
            for mode, head in (('base', base_head), ('adaptive_off', adaptive)):
                model.lm_head = head
                generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=24,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_id)
                outcomes[mode] = generated[0, prompt['input_ids'].shape[1]:].tolist()
        ordinary.append({'question': question, 'same_tokens': outcomes['base'] == outcomes['adaptive_off']})
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(source) != source_sha
            or any(base.sha(path) != layer_hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected input changed')
    OUT.mkdir()
    candidate = OUT / 'adaptive-head.pt'
    atomic_save({'format': 'spora-adaptive-semantic-head-v1', 'base_sha256': base.WEIGHT_SHA,
                 'writer_sha256': source_sha, 'layer_sha256': layer_hashes, 'rank': 32, 'gain': GAIN,
                 'state': {name: tensor.detach().cpu() for name, tensor in adaptive.state_dict().items()
                           if not name.startswith('base.')}}, candidate)
    summary = {mode: {'first_correct': sum(r['outputs'][mode]['first_correct'] for r in rows),
                      'ended': sum(r['outputs'][mode]['ended'] for r in rows),
                      'exact': sum(r['outputs'][mode]['exact'] for r in rows)} for mode in ('one_shot', 'adaptive')}
    report = {'summary': summary, 'rows': rows, 'ordinary': ordinary, 'trace': trace,
              'base_sha256': base.WEIGHT_SHA, 'writer_sha256': source_sha, 'layer_sha256': layer_hashes,
              'candidate_sha256': base.sha(candidate), 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Single-token synthetic facts and fixed question; oracle value spans and selected slots.',
                              'EOS is a training label but no EOS vector or step counter is used at generation.',
                              'Held values were used in previous project development; one seed and 16 generated cases.',
                              'No multi-token, natural routing, persistent individual or open dialogue validation.',
                              'Memory-off token comparison checks only three short examples.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
