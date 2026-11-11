"""Measure first-step token competition under two paraphrases and fixed facts."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora import integrated_open_base
from spora.memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor


def main():
    source = p.ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
    output = p.ROOT / 'runs/deepseek-spora-first-token-diagnosis-20261112.json'
    if output.exists():
        raise FileExistsError(output)
    candidate_sha = p.sha(source / 'adapter.pt')
    if candidate_sha != json.loads((source / 'result.json').read_text(encoding='utf-8'))['adapter_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = p.torch.load(source / 'adapter.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    for layer in (6, 12):
        layer_path = p.ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if payload['candidate_sha256'][str(layer)] != p.sha(layer_path):
            raise ValueError('Layer provenance mismatch')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    captures = []

    class Recorder(MemoryOutputLogitsProcessor):
        def __call__(self, input_ids, scores):
            before = scores.detach().float().clone()
            after = super().__call__(input_ids, scores).detach().float()
            if not captures:
                top = after[0].topk(8).indices.tolist()
                captures.append({'top_tokens': [{'id': i, 'text': tokenizer.decode([i]),
                                                 'base_score': float(before[0, i]),
                                                 'biased_score': float(after[0, i])} for i in top],
                                 'base_winner_id': int(before[0].argmax()),
                                 'biased_winner_id': int(after[0].argmax())})
            return after.to(scores.dtype)

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        original = p.frozen_digest(chat.organism, chat.state)
        decoder = chat.decoder.eval()
        old_head = decoder.output_adapter
        candidate = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
        candidate.load_state_dict(payload['state'], strict=True)
        decoder.output_adapter = candidate
        device = next(chat.organism.parameters()).device
        key, name, _ = p.ENROLLMENTS[1]
        for question in (f'{name}现在喜欢哪种饮料？', f'请说出{name}目前爱喝的饮品。'):
            inputs = p.closed_think_inputs(tokenizer, p.FILLER * 4 + question, 'cuda')
            length = inputs['input_ids'].shape[1]
            if not 128 < length <= 768 - p.MAX_NEW:
                raise ValueError('Unexpected prompt length')
            for value in (3, 4, 7):
                state = p.fresh_controlled_state(chat.organism)
                for fact in (1, value):
                    state, decision = p.observe_sourced(chat.organism, state, key, fact, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Controlled write rejected')
                before = p.frozen_digest(chat.organism, state)
                spec = p._memory_args(chat.organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                keys = p.torch.tensor([key], dtype=p.torch.long, device=device)
                with p.torch.no_grad():
                    logits = p.query(chat.organism, state, keys, p.torch.zeros_like(keys), p.torch.zeros_like(keys))
                captures.clear()
                with patch.object(integrated_open_base, 'MemoryOutputLogitsProcessor', Recorder), p.torch.inference_mode():
                    ids = decoder.generate(**inputs, **{**spec, 'use_cache': True}, memory_output_logits=logits,
                                           max_new_tokens=p.MAX_NEW, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                if len(captures) != 1 or p.frozen_digest(chat.organism, state) != before:
                    raise ValueError('Instrumentation or state invariant failed')
                tail = ids[0, length:].tolist()
                row = {'question': question, 'expected': p.WORDS[value],
                       'native_correct': int(logits.argmax(-1)[0]) == value,
                       'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                       'token_ids': tail, 'first_step': captures[0]}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        decoder.output_adapter = old_head
        if p.frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'candidate_sha256': candidate_sha, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Six development cases, one known entity and three seen values.',
                              'First-step scores show competition, not proof of a unique failure cause.',
                              'Addresses and memory evidence are provided by the experiment.',
                              'No default checkpoint or real individual modified.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
