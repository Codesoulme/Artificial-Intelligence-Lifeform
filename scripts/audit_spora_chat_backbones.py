"""Compare local chat backbones at the same generation budget."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest


DEVELOPMENT = ROOT / 'data/conversation-development-v2.json'
FILES = ('config.json', 'generation_config.json', 'model.safetensors',
         'tokenizer.json', 'tokenizer_config.json')


@torch.inference_mode()
def reply(model, tokenizer, history, question, *, device, deepseek, limit, eos, expected_prompt=None):
    messages = [*history, {'role': 'user', 'content': question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if deepseek:
        if not prompt.endswith('<think>\n'):
            raise ValueError('Unexpected DeepSeek template ending')
        prompt += '</think>\n\n'
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    if expected_prompt is not None and inputs.input_ids[0].tolist() != expected_prompt:
        raise ValueError('Generation prompt differs from the scored prompt')
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.generate(
        **inputs, use_cache=True, do_sample=False, max_new_tokens=limit,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    length = inputs.input_ids.shape[1]
    tokens = output[0, length:].tolist()
    answer = {'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
              'prompt_tokens': length, 'generated_tokens': len(tokens),
              'eos': bool(tokens and tokens[-1] in eos), 'seconds': seconds}
    if expected_prompt is not None:
        answer['token_ids'] = tokens
    return answer


def summarize(rows):
    return {'turns': len(rows), 'eos': sum(row['eos'] for row in rows),
            'median_seconds': median(row['seconds'] for row in rows),
            'truncated': sum(not row['eos'] for row in rows)}


def evaluate(model, tokenizer, development, families, settings):
    rollouts = {}
    for scenario in development:
        history, rows = [], []
        for question in scenario['turns']:
            row = reply(model, tokenizer, history, question, **settings)
            rows.append({'question': question, **row})
            history.extend([{'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': row['answer']}])
        rollouts[scenario['name']] = rows
        print(scenario['name'], sum(row['eos'] for row in rows), len(rows), flush=True)

    fixed = {}
    for family, cases in families.items():
        rows = []
        for case in cases:
            history = [{'role': role, 'content': text}
                       for user, answer in case['history']
                       for role, text in (('user', user), ('assistant', answer))]
            row = reply(model, tokenizer, history, case['question'], **settings)
            normalized = row['answer'].strip('。！!，,；;？?《》\n ')
            rows.append({'question': case['question'], 'preferred': case['preferred'],
                         'foil': case['foil'], 'history': case['history'], **row,
                         'strict': normalized == case['preferred'], 'stale': normalized == case['foil']})
        fixed[family] = rows
        print(family, dict(Counter('correct' if row['strict'] else 'stale' if row['stale'] else 'other'
                                   for row in rows)), flush=True)
    return rollouts, fixed


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    deepseek = args.qwen is None
    model_dir = MODEL if deepseek else args.qwen.resolve()
    sources = [DEVELOPMENT, *(model_dir / name for name in FILES)]
    sources.extend(path for path in (args.fixed, args.checkpoint, args.database) if path is not None)
    protected = {str(path): sha(path) for path in sources}
    development = json.loads(DEVELOPMENT.read_text(encoding='utf-8'))['scenarios']
    families = json.loads(args.fixed.read_text(encoding='utf-8'))['families'] if args.fixed else {}
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if deepseek:
        model = load_full_streamed()
        eos = [tokenizer.eos_token_id]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='sdpa',
        ).to('cuda').eval().requires_grad_(False)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
    settings = {'device': str(model.device), 'deepseek': deepseek, 'limit': args.max_new_tokens, 'eos': eos}
    with ExitStack() as resources:
        runtime = None
        if args.checkpoint:
            runtime = resources.enter_context(load_integrated_durable(args.database, args.checkpoint, model, MODEL))
            before = frozen_digest(runtime.organism, runtime.state)
            model = runtime.decoder
        reply(model, tokenizer, [], '你好。', **{**settings, 'limit': 8})
        rollouts, fixed = evaluate(model, tokenizer, development, families, settings)
        if runtime and frozen_digest(runtime.organism, runtime.state) != before:
            raise ValueError('Dialogue evaluation changed the loaded individual')
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('A benchmark source changed during the run')
    result = {
        'format': 'spora-chat-backbones-v1', 'model_dir': str(model_dir), 'sources': protected,
        'settings': settings, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.perf_counter() - started, 'rollouts': rollouts, 'fixed': fixed,
        'rollout_summary': summarize([row for rows in rollouts.values() for row in rows]),
        'fixed_summary': {family: {**summarize(rows), 'strict': sum(row['strict'] for row in rows),
                                   'stale': sum(row['stale'] for row in rows)}
                          for family, rows in fixed.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument('--qwen', type=Path, help='Local Qwen Instruct snapshot. Default is original DeepSeek.')
    model_options.add_argument('--checkpoint', type=Path, help='DeepSeek SPORA checkpoint with durable state.')
    parser.add_argument('--database', type=Path, help='Durable state for the SPORA checkpoint.')
    parser.add_argument('--fixed', type=Path, help='Prior SPORA constraint report providing fixed histories.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if bool(options.checkpoint) != bool(options.database):
        parser.error('--checkpoint and --database are required together')
    if options.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    main(options)
