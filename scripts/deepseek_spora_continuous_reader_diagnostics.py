"""Measure causal signal, token probabilities and gradients for the isolated reader."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_reader_pilot import WORDS
from spora.continuous_text_memory import ContinuousTextCodec

OUT = p.ROOT / 'runs/continuous-reader-diagnostics-20261112.json'
CANDIDATE = p.ROOT / 'runs/continuous-reader-generation-pilot-20261112/reader.pt'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    t = p.torch
    t.set_num_threads(2)
    t.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    candidate_sha = p.sha(CANDIDATE)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    ids = [tokenizer.encode(w, add_special_tokens=False) for w in WORDS]
    with tempfile.TemporaryDirectory() as tmp:
        p.runtime.WORKING_CHECKPOINT = Path(tmp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, organism = chat.decoder.eval(), chat.organism.eval()
        state_digest = p.frozen_digest(organism, chat.state)
        device = next(organism.parameters()).device
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=organism.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tokenizer.eos_token_id)
        decoder.requires_grad_(False)
        decoder.reader.requires_grad_(True)
        candidate = t.load(CANDIDATE, map_location='cpu', weights_only=True)
        if candidate['source_sha256'] != source_sha or candidate['base_sha256'] != p.WEIGHT_SHA:
            raise ValueError('Reader source mismatch')
        original = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        configs = [('trained_name_train_question', 0, '请只说出小林喜欢的水果名称。', (0, 1)),
                   ('new_name_train_question', 6, '请只说出小陈喜欢的水果名称。', (0, 1)),
                   ('trained_name_new_question', 0, '小林喜欢吃什么水果？只回答名称。', (0, 1)),
                   ('new_word_train_question', 0, '请只说出小林喜欢的水果名称。', (4, 5))]
        states = {}
        for _, key, _, values in configs:
            for value in values:
                if (key, value) in states:
                    continue
                state = p.fresh_controlled_state(organism)
                fast, velocity = organism.initial_state(1)
                target = t.zeros((1, organism.config.memory_width), device=device)
                code = codec.encode(t.tensor(ids[value], device=device))
                target[0, :code.numel()] = code
                for site in fast:
                    site[0] = t.zeros_like(site[0])
                with t.no_grad():
                    for i, site in enumerate(organism.memories):
                        fast[i], velocity[i] = site.write_temporal(fast[i], velocity[i],
                            site.current_basis[key:key + 1].to(device), target,
                            t.ones(1, device=device), t.zeros(1, dtype=t.long, device=device), False)
                state.current = [site[0].detach() for site in fast]
                states[key, value] = state

        rows = []
        for version, weights in [('original', original), ('trained', candidate['reader'])]:
            decoder.reader.load_state_dict(weights, strict=True)
            for label, key, question, values in configs:
                prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
                distributions = []
                for value in values:
                    spec = p._memory_args(organism, states[key, value],
                                          {'key': key, 'intent': 'current', 'event_time': 0})
                    measurements = []
                    def hook(module, args, result):
                        hidden, matrices, site, address = args
                        delta = result.detach().float() - hidden.detach().float()
                        native = module.native_sites[site]
                        with t.no_grad():
                            gate = t.sigmoid(native.read_gate(hidden.to(native.read_out.weight.dtype))).float()
                        measurements.append({'site': site, 'hidden_norm': float(hidden.detach().float().norm(dim=-1).mean()),
                            'residual_norm': float(delta.norm(dim=-1).mean()), 'gate_mean': float(gate.mean()),
                            'matrix_norm': float(matrices[0].float().norm())})
                    handle = decoder.reader.register_forward_hook(hook)
                    try:
                        with t.no_grad():
                            logits = decoder(**prompt, **spec).logits[0, -1].float()
                    finally:
                        handle.remove()
                    distributions.append(logits.cpu())
                    target = t.tensor([ids[value] + [tokenizer.eos_token_id]], device=device)
                    full = t.cat((prompt['input_ids'], target), 1)
                    decoder.zero_grad(set_to_none=True)
                    result = decoder(input_ids=full, attention_mask=t.ones_like(full), **spec)
                    prefix = prompt['input_ids'].shape[1]
                    scores = result.logits[:, prefix-1:prefix-1+target.shape[1]].float()
                    losses = t.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                                            target.reshape(-1), reduction='none')
                    losses.mean().backward()
                    grads = {name: float(param.grad.float().norm()) if param.grad is not None else None
                             for name, param in decoder.reader.named_parameters()
                             if 'read_gate' in name or 'read_out' in name}
                    rows.append({'version': version, 'split': label, 'word': WORDS[value],
                                 'target_ids': target[0].tolist(), 'token_nll': losses.detach().cpu().tolist(),
                                 'target_first_rank': int((logits > logits[ids[value][0]]).sum()) + 1,
                                 'readers': measurements, 'gradient_norms': grads})
                delta = (distributions[0] - distributions[1]).abs()
                for row in rows[-2:]:
                    row['pair_first_logit_max_abs_difference'] = float(delta.max())
                    row['pair_first_logit_rms_difference'] = float(delta.square().mean().sqrt())
            print(version, 'complete', flush=True)
        if p.frozen_digest(organism, chat.state) != state_digest:
            raise ValueError('Protected individual state changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(CANDIDATE) != candidate_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    report = {'rows': rows, 'source_sha256': source_sha, 'candidate_sha256': candidate_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - begin,
              'peak_allocated_bytes': t.cuda.max_memory_allocated(),
              'limitations': ['Teacher-forced diagnostics, not generated answers.',
                              'Four tiny authored contrasts with synthetic numeric addresses.',
                              'Read residual includes slow signal; it is not a pure measurement of the stored vector.',
                              'Only first-token logits are compared under two stored values.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
