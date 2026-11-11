"""Compare two fixed memory gains on fresh short paraphrases and held-out names."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from spora.memory_output import MemoryOutputAdapter


def main():
    source = p.ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
    output = p.ROOT / 'runs/deepseek-spora-gain-short-holdout-20261112.json'
    if output.exists():
        raise FileExistsError(output)
    candidate_sha = p.sha(source / 'adapter.pt')
    if candidate_sha != json.loads((source / 'result.json').read_text(encoding='utf-8'))['adapter_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = p.torch.load(source / 'adapter.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    for layer in (6, 12):
        path = p.ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if payload['candidate_sha256'][str(layer)] != p.sha(path):
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
        for index in (6, 7):
            key, name = KEYS[index], NAMES[index]
            for template in ('提到饮料时，{name}如今会选什么？', '{name}的饮品喜好现在是什么？'):
                question = template.format(name=name)
                inputs = p.closed_think_inputs(tokenizer, question, 'cuda')
                length = inputs['input_ids'].shape[1]
                for value in (3, 4):
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
                    native = int(logits.argmax(-1)[0])
                    results = {}
                    for label, gain in (('default', 1.0), ('stronger', 1.3)):
                        with p.torch.inference_mode():
                            ids = decoder.generate(**inputs, **{**spec, 'use_cache': True}, memory_output_logits=logits,
                                memory_output_gain=gain, max_new_tokens=p.MAX_NEW, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                        tail = ids[0, length:].tolist()
                        answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                        results[label] = {'answer': answer, 'token_ids': tail,
                                          'exact': answer.rstrip('。.!！ ') == p.WORDS[value],
                                          'ended_eos': tokenizer.eos_token_id in tail}
                    if p.frozen_digest(chat.organism, state) != before:
                        raise ValueError('Temporary state changed')
                    row = {'name': name, 'key': key, 'question': question, 'input_tokens': length,
                           'expected': p.WORDS[value], 'native_correct': native == value, 'results': results}
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=True), flush=True)
        decoder.output_adapter = old_head
        if p.frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    summary = {'total': len(rows), 'native_correct': sum(r['native_correct'] for r in rows)}
    for label in ('default', 'stronger'):
        summary[label] = {'exact': sum(r['results'][label]['exact'] for r in rows),
                          'ended_eos': sum(r['results'][label]['ended_eos'] for r in rows)}
    result = {'summary': summary, 'rows': rows, 'candidate_sha256': candidate_sha,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Two names held out of adapter training, two freshly authored questions, two trained values.',
                              'Gain 1.3 was chosen from development data; this is a small post-selection test.',
                              'Addresses and evidence supplied by test; no natural-language routing.',
                              'Exact matching can miss valid longer responses; inspect actual outputs.',
                              'No default policy or actual individual changed.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
