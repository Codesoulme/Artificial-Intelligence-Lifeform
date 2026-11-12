"""Audit answer-token alignment and conflicting memory supervision gradients."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_reader_full_answer_pilot import unique_token_span
from spora.continuous_text_memory import ContinuousTextCodec

OUT = p.ROOT / 'runs/continuous-fact-gradient-audit-20261112.json'
CANDIDATE = p.ROOT / 'runs/continuous-reader-broader-fact-pilot-20261112/reader.pt'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    t = p.torch
    t.set_num_threads(2)
    t.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha, candidate_sha = p.sha(p.runtime.CHECKPOINT), p.sha(CANDIDATE)
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    words = ('苹果', '香蕉')
    word_ids = [tok.encode(w, add_special_tokens=False) for w in words]
    if any(len(ids) != 1 for ids in word_ids) or word_ids[0] == word_ids[1]:
        raise ValueError('Distinct single-token fruit words required')
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        p.runtime.WORKING_CHECKPOINT = Path(tmp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, organism = chat.decoder.eval(), chat.organism.eval()
        source_digest = p.frozen_digest(organism, chat.state)
        device = next(organism.parameters()).device
        candidate = t.load(CANDIDATE, map_location='cpu', weights_only=True)
        if (candidate['source_sha256'] != source_sha or candidate['base_sha256'] != p.WEIGHT_SHA
                or not candidate.get('requires_slow_disabled') or not candidate.get('requires_zero_output_bias')):
            raise ValueError('Candidate provenance or research conditions mismatch')
        decoder.reader.load_state_dict(candidate['reader'], strict=True)
        decoder.requires_grad_(False)
        decoder.reader.requires_grad_(True)
        for site in decoder.reader.native_sites:
            if not bool((site.read_out.bias == 0).all()):
                raise ValueError('Expected zero output bias')
            site.read_out.bias.requires_grad_(False)
        params = [s.read_out.weight for s in decoder.reader.native_sites]
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=organism.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tok.eos_token_id)
        patches = [patch.object(site, 'slow_signal', side_effect=lambda x, query: t.zeros_like(query))
                   for site in decoder.reader.native_sites]
        for pa in patches:
            pa.start()
        try:
            states = {}
            for key in (0, 6):
                for value in (0, 1):
                    state = p.fresh_controlled_state(organism)
                    fast, velocity = organism.initial_state(1)
                    for site in fast:
                        site[0] = t.zeros_like(site[0])
                    code = codec.encode(t.tensor(word_ids[value], device=device))
                    target = t.zeros((1, organism.config.memory_width), device=device)
                    target[0, :code.numel()] = code
                    with t.no_grad():
                        for i, site in enumerate(organism.memories):
                            fast[i], velocity[i] = site.write_temporal(fast[i], velocity[i],
                                site.current_basis[key:key+1].to(device), target,
                                t.ones(1, device=device), t.zeros(1, dtype=t.long, device=device), False)
                    state.current = [x[0].detach() for x in fast]
                    states[key, value] = state
            state_hashes = {key: p.frozen_digest(organism, value) for key, value in states.items()}
            for key, name, template in ((0, '小林', '请只说出{name}喜欢的水果名称。'),
                                        (6, '小陈', '请只说出{name}喜欢的水果名称。'),
                                        (0, '小林', '{name}爱吃什么水果？只回答名字。')):
                question = template.format(name=name)
                prompt = p.closed_think_inputs(tok, question, 'cuda')
                answer_prefix = f'{name}喜欢的水果是'
                prefix_ids = tok.encode(answer_prefix, add_special_tokens=False)
                full_answer = tok.encode(answer_prefix + words[0] + '。', add_special_tokens=False)
                position = unique_token_span(full_answer, word_ids[0])
                if full_answer[:position] != prefix_ids:
                    raise ValueError('Tokenization mismatch between prefix and complete answer')
                prefill = t.cat((prompt['input_ids'], t.tensor([prefix_ids], device=device)), dim=1)
                full = t.cat((prompt['input_ids'], t.tensor([full_answer + [tok.eos_token_id]], device=device)), dim=1)
                gradient_list = []
                per_value = []
                for value in (0, 1):
                    spec = p._memory_args(organism, states[key, value],
                                          {'key': key, 'intent': 'current', 'event_time': 0})
                    result = decoder(input_ids=full, attention_mask=t.ones_like(full), **spec).logits.float()
                    fact_logits = result[0, prompt['input_ids'].shape[1] - 1 + position]
                    with t.no_grad():
                        short_logits = decoder(input_ids=prefill, attention_mask=t.ones_like(prefill), **spec).logits[0, -1].float()
                    target_id = word_ids[value][0]
                    loss = t.nn.functional.cross_entropy(fact_logits[None, :], t.tensor([target_id], device=device))
                    grads = t.autograd.grad(loss, params, allow_unused=False)
                    gradient_list.append([x.detach().float() for x in grads])
                    diff = (fact_logits.detach() - short_logits).abs()
                    per_value.append({'word': words[value], 'target_nll': float(loss.detach()),
                        'target_rank': int((fact_logits.detach() > fact_logits.detach()[target_id]).sum()) + 1,
                        'apple_minus_banana': float(fact_logits.detach()[word_ids[0][0]] - fact_logits.detach()[word_ids[1][0]]),
                        'prefix_logit_max_abs_diff': float(diff.max()), 'prefix_logit_rms_diff': float(diff.square().mean().sqrt()),
                        'target_logit_abs_diff': float(diff[target_id]),
                        'gradient_norms': [float(g.norm()) for g in grads]})
                comparisons = []
                for a, b in zip(*gradient_list):
                    na, nb = float(a.norm()), float(b.norm())
                    comparisons.append({'cosine': float(t.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                                        'sum_norm_over_sum_norms': float((a+b).norm()) / (na+nb) if na+nb else None})
                rows.append({'key': key, 'name': name, 'question': question, 'fact_position': position,
                             'answer_prefix_tokens': prefix_ids, 'per_value': per_value,
                             'per_site_gradients': comparisons})
                print(json.dumps({'key': key, 'question': question, 'per_value': per_value,
                                  'per_site_gradients': comparisons}, ensure_ascii=True), flush=True)
            if p.frozen_digest(organism, chat.state) != source_digest or any(
                    p.frozen_digest(organism, state) != state_hashes[key] for key, state in states.items()):
                raise ValueError('Protected state changed')
        finally:
            for pa in patches:
                pa.stop()
    if (p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(CANDIDATE) != candidate_sha
            or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA):
        raise ValueError('Protected files changed')
    result = {'rows': rows, 'source_sha256': source_sha, 'candidate_sha256': candidate_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': t.cuda.max_memory_allocated(),
              'limitations': ['Three authored questions, two words, fixed synthetic numeric addresses.',
                              'Teacher-forced fact-token loss; not generated answers or full paired objective.',
                              'Gradient cosine is only for read_out weights, not gates or all reader parameters.',
                              'FP16 forward arithmetic can cause small sequence-length-dependent differences.',
                              'No parameter update or individual modification.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
