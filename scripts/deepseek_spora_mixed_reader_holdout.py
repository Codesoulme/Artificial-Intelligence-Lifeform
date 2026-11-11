"""Probe an isolated reader candidate on fresh questions and misuse controls."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES


def main():
    source = p.ROOT / 'runs/deepseek-spora-reader-mixed-answer-20261112'
    output = p.ROOT / 'runs/deepseek-spora-mixed-reader-holdout-20261112.json'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = p.sha(source / 'reader.pt')
    if candidate_sha != report['candidate_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = p.torch.load(source / 'reader.pt', map_location='cpu', weights_only=True)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if payload['source_checkpoint_sha256'] != checkpoint_sha or payload['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Candidate provenance mismatch')
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        decoder, organism = chat.decoder.eval(), chat.organism.eval()
        initial = p.frozen_digest(organism, chat.state)
        old = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        heads = (('old', old), ('candidate', payload['state']))
        device = next(organism.parameters()).device

        def generate(question, state, key):
            prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
            length = prompt['input_ids'].shape[1]
            spec = {} if state is None else p._memory_args(organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
            spec['use_cache'] = True
            with p.torch.inference_mode():
                ids = decoder.generate(**prompt, **spec, max_new_tokens=24, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, length:].tolist()
            return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                    'token_ids': tail, 'ended_eos': tokenizer.eos_token_id in tail}

        for index in (6, 7):
            key, name = KEYS[index], NAMES[index]
            for template in ('按照记忆，{name}现在偏好什么饮品？', '如今{name}一般爱喝哪一种饮料？'):
                question = template.format(name=name)
                for value in (3, 4):
                    state = p.fresh_controlled_state(organism)
                    for fact in (1, value):
                        state, decision = p.observe_sourced(organism, state, key, fact, 'direct', 1.0)
                        if not decision['current_write']:
                            raise ValueError('Controlled write rejected')
                    before = p.frozen_digest(organism, state)
                    keys = p.torch.tensor([key], dtype=p.torch.long, device=device)
                    with p.torch.no_grad():
                        evidence = p.query(organism, state, keys, p.torch.zeros_like(keys), p.torch.zeros_like(keys))
                    native = int(evidence.argmax(-1)[0])
                    results = {}
                    for label, weights in heads:
                        decoder.reader.load_state_dict(weights, strict=True)
                        results[label] = generate(question, state, key)
                        results[label]['exact'] = results[label]['answer'].rstrip('。.!！ ') == p.WORDS[value]
                    if p.frozen_digest(organism, state) != before:
                        raise ValueError('Temporary state changed')
                    row = {'group': 'fact', 'key': key, 'question': question, 'expected': p.WORDS[value],
                           'native_correct': native == value, 'results': results}
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=True), flush=True)

        key, other = KEYS[6], KEYS[7]
        empty = p.fresh_controlled_state(organism)
        unrelated = p.fresh_controlled_state(organism)
        unrelated, decision = p.observe_sourced(organism, unrelated, other, 4, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Unrelated memory write rejected')
        negative_before = [p.frozen_digest(organism, s) for s in (empty, unrelated)]
        for group, question in (('unknown_fact', f'{NAMES[6]}现在爱喝什么饮料？'),
                                ('ordinary', '请写一句鼓励努力学习的话。')):
            results = {}
            for label, weights in heads:
                decoder.reader.load_state_dict(weights, strict=True)
                for condition, state in (('no_memory', None), ('empty_memory', empty), ('unrelated_memory', unrelated)):
                    results[label + '_' + condition] = generate(question, state, key)
            row = {'group': group, 'question': question, 'results': results}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        if [p.frozen_digest(organism, s) for s in (empty, unrelated)] != negative_before:
            raise ValueError('Negative-control state changed')
        decoder.reader.load_state_dict(old, strict=True)
        if p.frozen_digest(organism, chat.state) != initial:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    facts = [row for row in rows if row['group'] == 'fact']
    summary = {'facts': len(facts), 'native_correct': sum(row['native_correct'] for row in facts)}
    for label in ('old', 'candidate'):
        summary[label] = {'exact': sum(row['results'][label]['exact'] for row in facts),
                          'ended_eos': sum(row['results'][label]['ended_eos'] for row in facts),
                          'both_values_correct': sum(all(row['results'][label]['exact'] for row in facts if row['question'] == q)
                              for q in {row['question'] for row in facts})}
    result = {'summary': summary, 'rows': rows, 'candidate_sha256': candidate_sha,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Two held-out names, two newly authored questions and two seen finite values.',
                              'Known person was present in prior development evaluation.',
                              'All addresses supplied by test; no natural-language routing or write.',
                              'Negative controls deliberately force memory reading, not actual runtime gate behavior.',
                              'Exact match can undercount valid sentences; inspect all answers.',
                              'No default reader or real individual changed.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
