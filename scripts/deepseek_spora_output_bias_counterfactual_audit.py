"""Compare finite-memory output adapters and their biases on controlled changes."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-output-bias-counterfactual-20261114.json'
CASES = (
    (0, '请告诉我小林目前想喝哪种饮料？', (6, 7)),
    (3, '对于阿哲，记录里现在的饮料是什么？', (3, 4)),
    (2, '查一下小雨保存的饮料喜好。', (2, 3)),
    (6, '从你记下的内容看，{name}平时偏爱哪种饮品？', (1, 5)),
    (7, '我想确认一下，{name}喜欢喝的是什么？', (0, 4)),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    candidate_sha = p.sha(SOURCE / 'adapter.pt')
    if base_sha != p.WEIGHT_SHA or candidate_sha != json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))['adapter_sha256']:
        raise ValueError('Base or candidate checksum mismatch')
    payload = torch.load(SOURCE / 'adapter.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != base_sha:
        raise ValueError('Candidate base mismatch')
    for layer in (6, 12):
        layer_path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if payload['candidate_sha256'][str(layer)] != p.sha(layer_path):
            raise ValueError('Candidate layer mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-output-bias-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        old = chat.decoder.output_adapter
        new = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
        new.load_state_dict(payload['state'], strict=True)
        adapters = {'current': old, 'v2': new}
        for index, template, targets in CASES:
            question = template.format(name=NAMES[index])
            key = KEYS[index]
            state = p.fresh_controlled_state(chat.organism)
            for target in targets:
                state, decision = p.observe_sourced(chat.organism, state, key, target, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                state_digest = p.frozen_digest(chat.organism, state)
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                native = int(evidence.argmax(-1)[0])
                prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
                output = {}
                for label in ('none', 'current', 'v2'):
                    adapter = adapters.get(label)
                    if adapter is not None:
                        chat.decoder.output_adapter = adapter
                        with torch.inference_mode():
                            bias = adapter(evidence).float()[0]
                        top_values, top_ids = bias.topk(5)
                        bias_info = [{'id': int(token), 'token': chat.tokenizer.decode([int(token)]),
                                      'bias': float(value)} for token, value in zip(top_ids, top_values)]
                        target_ids = chat.tokenizer.encode(p.WORDS[target], add_special_tokens=False)
                        target_bias = float(bias[target_ids[0]]) if target_ids else None
                    else:
                        bias_info, target_bias = [], None
                    with torch.inference_mode():
                        ids = chat.decoder.generate(**prompt, **args,
                                                    memory_output_logits=evidence if adapter is not None else None,
                                                    max_new_tokens=32, do_sample=False,
                                                    pad_token_id=chat.tokenizer.pad_token_id,
                                                    eos_token_id=chat.tokenizer.eos_token_id)
                    tail = ids[0, prompt['input_ids'].shape[1]:].tolist()
                    answer = chat.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    output[label] = {'answer': answer, 'exact': answer.rstrip('。.!！ ') == p.WORDS[target],
                                     'token_ids': tail, 'target_first_token_bias': target_bias,
                                     'top_bias_tokens': bias_info}
                if p.frozen_digest(chat.organism, state) != state_digest:
                    raise ValueError('Temporary state changed during generation')
                row = {'question': question, 'key': key, 'expected': p.WORDS[target],
                       'target_id': target, 'native_id': native, 'results': output}
                rows.append(row)
                print(json.dumps({'done': len(rows), 'native_correct': native == target,
                                  'answers': {k: v['answer'] for k, v in output.items()}}, ensure_ascii=True), flush=True)
        chat.decoder.output_adapter = old
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(SOURCE / 'adapter.pt') != candidate_sha:
        raise ValueError('Protected file changed')
    summary = {'cases': len(rows), 'native_correct': sum(r['native_id'] == r['target_id'] for r in rows)}
    for label in ('none', 'current', 'v2'):
        summary[label] = {'exact': sum(r['results'][label]['exact'] for r in rows)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'candidate_sha256': candidate_sha, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small development set with reused and newly authored questions; not an independent test.',
                              'Some people or values may overlap training; values are from a finite learned vocabulary.',
                              'Direct numerical address bypasses language routing and enrollment.',
                              'Exact matching misses some valid paraphrases; inspect individual responses.',
                              'Top first-token bias does not determine output independently of model logits.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
