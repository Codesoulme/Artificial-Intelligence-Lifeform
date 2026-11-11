"""Controlled counterfactual values for identical long-prompt questions."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.memory_output import MemoryOutputAdapter


def main():
    source = p.ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
    output = p.ROOT / 'runs/deepseek-spora-long-memory-counterfactual-20261112.json'
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
        base, decoder = chat.base.eval(), chat.decoder.eval()
        if any(x.is_meta for x in list(base.parameters()) + list(base.buffers())):
            raise ValueError('Unmaterialized layer')
        original_digest = p.frozen_digest(chat.organism, chat.state)
        old_head = decoder.output_adapter
        new_head = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
        new_head.load_state_dict(payload['state'], strict=True)
        device = next(chat.organism.parameters()).device
        key, name, _ = p.ENROLLMENTS[1]
        questions = (f'{name}现在喜欢哪种饮料？', f'请说出{name}目前爱喝的饮品。')
        for question in questions:
            inputs = p.closed_think_inputs(tokenizer, p.FILLER * 4 + question, 'cuda')
            length = inputs['input_ids'].shape[1]
            if not 128 < length <= 768 - p.MAX_NEW:
                raise ValueError('Unexpected prompt length')
            for value in (3, 4, 6, 7):
                state = p.fresh_controlled_state(chat.organism)
                for written in (1, value):
                    state, decision = p.observe_sourced(chat.organism, state, key, written, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Synthetic write rejected')
                before = p.frozen_digest(chat.organism, state)
                spec = p._memory_args(chat.organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                keys = p.torch.tensor([key], dtype=p.torch.long, device=device)
                with p.torch.no_grad():
                    logits = p.query(chat.organism, state, keys, p.torch.zeros_like(keys), p.torch.zeros_like(keys))
                native = int(logits.argmax(-1)[0])
                results = {}
                for label, head in (('old', old_head), ('v2', new_head)):
                    decoder.output_adapter = head
                    with p.torch.inference_mode():
                        ids = decoder.generate(**inputs, **{**spec, 'use_cache': True}, memory_output_logits=logits,
                            max_new_tokens=p.MAX_NEW, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                    tokens = ids[0, length:].tolist()
                    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    results[label] = {'answer': answer, 'token_ids': tokens,
                                      'exact': answer.rstrip('。.!！ ') == p.WORDS[value]}
                if p.frozen_digest(chat.organism, state) != before:
                    raise ValueError('Temporary state changed')
                row = {'question': question, 'key': key, 'expected': p.WORDS[value],
                       'native_top_value': p.WORDS[native], 'native_correct': native == value,
                       'input_tokens': length, 'results': results}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        decoder.output_adapter = old_head
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {'native_correct': sum(r['native_correct'] for r in rows), 'total': len(rows)}
    for label in ('old', 'v2'):
        summary[label] = {'exact': sum(r['results'][label]['exact'] for r in rows),
                          'all_four_values_correct_by_question': sum(all(r['results'][label]['exact'] for r in rows if r['question'] == q) for q in questions)}
    result = {'summary': summary, 'rows': rows, 'candidate_sha256': candidate_sha, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['One known entity, two authored questions and four seen drink values.',
                              'Address and current-value query are provided by the test.',
                              'Independent temporary states, not sequential natural dialogue.',
                              'Exact match misses semantically valid longer answers; inspect rows.',
                              'No default checkpoint or real individual modified.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
