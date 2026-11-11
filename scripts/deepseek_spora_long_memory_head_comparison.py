"""Compare output heads on the same controlled long-prompt memory snapshots."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as probe
from spora.memory_output import MemoryOutputAdapter


def main():
    p = probe
    source = p.ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
    output = p.ROOT / 'runs/deepseek-spora-long-memory-head-comparison-20261112.json'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = p.sha(source / 'adapter.pt')
    if candidate_sha != report['adapter_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = p.torch.load(source / 'adapter.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Candidate base mismatch')
    for index in (6, 12):
        layer = p.ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{index}.pt'
        if payload['candidate_sha256'][str(index)] != p.sha(layer):
            raise ValueError('Candidate layer provenance mismatch')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    prior = json.loads(p.OUTPUT.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior checkpoint mismatch')
    previous = {(r['key'], r['mode']): r for r in prior['rows']}
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        base, decoder = chat.base.eval(), chat.decoder.eval()
        if any(x.is_meta for x in list(base.parameters()) + list(base.buffers())):
            raise ValueError('Unmaterialized layer')
        original_digest = p.frozen_digest(chat.organism, chat.state)
        state = p.fresh_controlled_state(chat.organism)
        for person, old, new in p.FACTS:
            for value in (old, new):
                state, decision = p.observe_sourced(chat.organism, state, p.ENROLLMENTS[person][0], value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Synthetic write rejected')
        state_digest = p.frozen_digest(chat.organism, state)
        old_head = decoder.output_adapter
        new_head = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
        new_head.load_state_dict(payload['state'], strict=True)
        device = next(chat.organism.parameters()).device
        for index, (person, old, new) in enumerate(p.FACTS):
            key, name, _ = p.ENROLLMENTS[person]
            for mode, target in (('current', new), ('past', old)):
                question = f'{name}现在喜欢哪种饮料？' if mode == 'current' else f'{name}以前喜欢哪种饮料？'
                inputs = p.closed_think_inputs(tokenizer, p.FILLER * 4 + question, 'cuda')
                length = inputs['input_ids'].shape[1]
                if not (128 < length and length + p.MAX_NEW <= 768):
                    raise ValueError('Unexpected prompt length')
                spec = p._memory_args(chat.organism, state, {'key': key, 'intent': mode, 'event_time': index * 2})
                keys = p.torch.tensor([key], dtype=p.torch.long, device=device)
                modes = p.torch.zeros_like(keys) if mode == 'current' else p.torch.ones_like(keys)
                times = p.torch.zeros_like(keys) if mode == 'current' else p.torch.full_like(keys, index * 2)
                with p.torch.no_grad():
                    logits = p.query(chat.organism, state, keys, modes, times)
                if int(logits.argmax(-1)[0]) != target:
                    raise ValueError('Native evidence mismatch')
                results = {}
                for label, head, evidence in (('old', old_head, logits), ('v2', new_head, logits), ('no_bias', old_head, None)):
                    decoder.output_adapter = head
                    with patch.object(decoder.reader, 'forward', wraps=decoder.reader.forward) as reader, p.torch.inference_mode():
                        ids = decoder.generate(**inputs, **{**spec, 'use_cache': True}, memory_output_logits=evidence,
                            max_new_tokens=p.MAX_NEW, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    if reader.call_count < 2:
                        raise ValueError('Reader sites not invoked')
                    tokens = ids[0, length:].tolist()
                    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    results[label] = {'answer': answer, 'token_ids': tokens, 'exact': answer == p.WORDS[target],
                                      'exact_ignoring_terminal_punctuation': answer.rstrip('。.!！ ') == p.WORDS[target]}
                reproduces = results['old']['token_ids'] == previous[(key, mode)]['outputs']['cached']['token_ids']
                rows.append({'key': key, 'mode': mode, 'expected': p.WORDS[target], 'input_tokens': length,
                             'old_reproduces_prior': reproduces, 'results': results})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        decoder.output_adapter = old_head
        if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Individual state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {label: {'exact': sum(r['results'][label]['exact'] for r in rows),
                       'punctuation_normalized': sum(r['results'][label]['exact_ignoring_terminal_punctuation'] for r in rows)}
               for label in ('old', 'v2', 'no_bias')}
    result = {'summary': summary, 'rows': rows, 'candidate_sha256': candidate_sha,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'prior_sha256': p.sha(p.OUTPUT), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously observed cases with repeated synthetic filler, not an independent test.',
                              'Addresses, times and evidence supplied by test; limited to finite drink values.',
                              'Exact metrics omit semantic equivalents and do not assess unsupported details.',
                              'No checkpoint, default adapter or real individual changed.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
