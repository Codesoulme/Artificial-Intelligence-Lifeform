"""Ablate two reader sites with fixed question, answer prefix and memory values."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.continuous_text_memory import ContinuousTextCodec

OUT = p.ROOT / 'runs/continuous-reader-site-causal-audit-20261112.json'
CANDIDATE = p.ROOT / 'runs/continuous-reader-full-teacher-control-20261112/reader.pt'
WORDS = ('苹果', '香蕉')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    source_sha, candidate_sha = p.sha(p.runtime.CHECKPOINT), p.sha(CANDIDATE)
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    word_ids = [tok.encode(word, add_special_tokens=False) for word in WORDS]
    if any(len(ids) != 1 for ids in word_ids) or word_ids[0] == word_ids[1]:
        raise ValueError('This diagnostic requires distinct one-token targets')
    with tempfile.TemporaryDirectory() as tmp:
        p.runtime.WORKING_CHECKPOINT = Path(tmp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model = chat.decoder.eval(), chat.organism.eval()
        original_digest = p.frozen_digest(model, chat.state)
        old_reader = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        saved = torch.load(CANDIDATE, map_location='cpu', weights_only=True)
        if saved['source_sha256'] != source_sha or saved['base_sha256'] != p.WEIGHT_SHA:
            raise ValueError('Candidate source mismatch')
        device = next(model.parameters()).device
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=model.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tok.eos_token_id)
        states = {}
        for key in (0, 6):
            for value in range(2):
                state = p.fresh_controlled_state(model)
                fast, velocity = model.initial_state(1)
                target = torch.zeros((1, model.config.memory_width), device=device)
                code = codec.encode(torch.tensor(word_ids[value], device=device))
                target[0, :code.numel()] = code
                for site in fast:
                    site[0] = torch.zeros_like(site[0])
                with torch.no_grad():
                    for i, site in enumerate(model.memories):
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], site.current_basis[key:key+1].to(device),
                            target, torch.ones(1, device=device),
                            torch.zeros(1, dtype=torch.long, device=device), False)
                state.current = [x[0].detach() for x in fast]
                states[key, value] = state
        digests = {k: p.frozen_digest(model, v) for k, v in states.items()}
        rows = []
        default_strength = decoder.reader.strength
        for version, weights in (('original', old_reader), ('trained', saved['reader'])):
            decoder.reader.load_state_dict(weights, strict=True)
            for strength in sorted(set((default_strength, 1.0))):
                decoder.reader.strength = strength
                for key, name in ((0, '小林'), (6, '小陈')):
                    question = f'请只说出{name}喜欢的水果名称。'
                    inputs = p.closed_think_inputs(tok, question, 'cuda')
                    # Supply identical response prefix, stopping immediately before the fact.
                    prefix = tok(f'{name}喜欢的水果是', add_special_tokens=False, return_tensors='pt').input_ids.to(device)
                    tokens = torch.cat((inputs['input_ids'], prefix), dim=1)
                    for active in ((), (0,), (1,), (0, 1)):
                        for value in range(2):
                            spec = p._memory_args(model, states[key, value],
                                {'key': key, 'intent': 'current', 'event_time': 0})
                            calls = []
                            original_forward = decoder.reader.forward
                            def ablate(hidden, matrices, site, address=None):
                                calls.append(site)
                                if site not in active:
                                    return hidden
                                return original_forward(hidden, matrices, site, address)
                            with patch.object(decoder.reader, 'forward', side_effect=ablate), torch.inference_mode():
                                logits = decoder(input_ids=tokens, attention_mask=torch.ones_like(tokens), **spec).logits[0, -1].float()
                            if calls != [0, 1]:
                                raise ValueError(f'Unexpected reader calls {calls}')
                            logprobs = logits.log_softmax(-1)
                            rows.append({'version': version, 'strength': strength, 'name': name, 'key': key,
                                         'active_sites': list(active), 'stored': WORDS[value],
                                         'apple_minus_banana': float(logits[word_ids[0][0]]-logits[word_ids[1][0]]),
                                         'target_logprob': float(logprobs[word_ids[value][0]]),
                                         'target_rank': int((logits > logits[word_ids[value][0]]).sum())+1})
            print(version, 'complete', flush=True)
        decoder.reader.strength = default_strength
        if p.frozen_digest(model, chat.state) != original_digest or any(
                p.frozen_digest(model, state) != digests[key] for key, state in states.items()):
            raise ValueError('Protected state changed')
    if (p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(CANDIDATE) != candidate_sha
            or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA):
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'default_strength': default_strength, 'source_sha256': source_sha,
              'candidate_sha256': candidate_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter()-begin, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Teacher-forced fixed prefix; no actual text generation in this audit.',
                              'Two human-authored values and two persons, with numeric addresses.',
                              'Site ablation includes both fast and slow paths within the active reader.',
                              'Strength 1 is an inference-only intervention on a reader trained at its default strength.',
                              'Positive apple-minus-banana favours apple; difference between memories quantifies sensitivity, not accuracy.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'seconds': result['seconds'], 'default_strength': default_strength}), flush=True)


if __name__ == '__main__':
    main()
