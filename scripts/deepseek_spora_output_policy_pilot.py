"""Compare isolated output-bias policies, including deliberate misuse controls."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora import integrated_open_base
from spora.memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor


class PersistentBias(MemoryOutputLogitsProcessor):
    def __call__(self, input_ids, scores):
        self._applied = False
        return super().__call__(input_ids, scores)


def main():
    source = p.ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
    output = p.ROOT / 'runs/deepseek-spora-output-policy-pilot-20261112.json'
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
        factual_question = f'请说出{name}目前爱喝的饮品。'
        cases = [('fact', 3, factual_question), ('fact', 4, factual_question),
                 ('fact', 7, factual_question), ('misrouted_nonmemory', 4, '请写一句鼓励努力学习的话。'),
                 ('wrong_evidence', 4, factual_question)]
        for group, value, question in cases:
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
                historical = p.query(chat.organism, state, keys, p.torch.ones_like(keys), p.torch.zeros_like(keys))
            if int(logits.argmax(-1)[0]) != value or int(historical.argmax(-1)[0]) != 1:
                raise ValueError('Native evidence mismatch')
            evidence = historical if group == 'wrong_evidence' else logits
            inputs = p.closed_think_inputs(tokenizer, p.FILLER * 4 + question, 'cuda')
            length = inputs['input_ids'].shape[1]
            if not 128 < length <= 768 - p.MAX_NEW:
                raise ValueError('Unexpected prompt length')
            results = {}
            if group == 'misrouted_nonmemory':
                with p.torch.inference_mode():
                    ids = decoder.generate(**inputs, use_cache=True, max_new_tokens=p.MAX_NEW, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tail = ids[0, length:].tolist()
                results['no_memory'] = {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                                        'token_ids': tail, 'ended_eos': tokenizer.eos_token_id in tail}
            policies = (('first_default', 1.0, MemoryOutputLogitsProcessor),
                        ('first_stronger', 1.3, MemoryOutputLogitsProcessor),
                        ('persistent', 1.0, PersistentBias))
            for label, gain, processor in policies:
                with patch.object(integrated_open_base, 'MemoryOutputLogitsProcessor', processor), p.torch.inference_mode():
                    ids = decoder.generate(**inputs, **{**spec, 'use_cache': True}, memory_output_logits=evidence,
                        memory_output_gain=gain, max_new_tokens=p.MAX_NEW, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tail = ids[0, length:].tolist()
                answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                results[label] = {'answer': answer, 'token_ids': tail,
                                  'ended_eos': tokenizer.eos_token_id in tail,
                                  'exact_current': answer.rstrip('。.!！ ') == p.WORDS[value]}
            if p.frozen_digest(chat.organism, state) != before:
                raise ValueError('Temporary state changed')
            row = {'group': group, 'question': question, 'actual_current': p.WORDS[value],
                   'evidence_top': p.WORDS[int(evidence.argmax(-1)[0])], 'results': results}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        decoder.output_adapter = old_head
        if p.frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    report = {'rows': rows, 'candidate_sha256': candidate_sha, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Three known positive cases, one forced non-memory misroute, one forced stale evidence case.',
                              'Deliberate misuse controls are not actual router behavior.',
                              'All evidence and numeric addresses supplied by the test.',
                              'Not an independent benchmark; no default policy or real individual modified.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
