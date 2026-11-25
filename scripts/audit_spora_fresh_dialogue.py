"""Compare local backbones on frozen Chinese dialogues of three to five turns."""

import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

from scripts.spora_fresh_dialogue_cases import CASES


EARLY_BUDGET = 96
FINAL_BUDGET = 224
POSITIONS = 1024
QWEN35_REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
QWEN35_WEIGHTS = 'model.safetensors-00001-of-00001.safetensors'
QWEN35_HASHES = {
    QWEN35_WEIGHTS: '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def evaluate(answer):
    rows = []
    for case in CASES:
        history = []
        rollout = []
        for index, question in enumerate(case.turns, start=1):
            budget = FINAL_BUDGET if index == len(case.turns) else EARLY_BUDGET
            result = answer(history, question, budget)
            rollout.append(result)
            history.append((question, result['answer']))
        resolved = answer([], case.resolved_prompt, FINAL_BUDGET)
        rows.append({**asdict(case), 'rollout': rollout, 'resolved': resolved})
        print(case.name, len(rollout), rollout[-1]['eos'], resolved['eos'], flush=True)
    turns = [turn for row in rows for turn in row['rollout']]
    return {
        'rows': rows,
        'turns': len(turns),
        'turns_ended': sum(turn['eos'] for turn in turns),
        'finals_ended': sum(row['rollout'][-1]['eos'] for row in rows),
        'resolved_ended': sum(row['resolved']['eos'] for row in rows),
    }


def qwen35_answer(model, tokenizer, history, question, budget):
    import torch

    messages = [
        {'role': role, 'content': text}
        for user, assistant in history
        for role, text in (('user', user), ('assistant', assistant))
    ]
    messages.append({'role': 'user', 'content': question})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    length = inputs.input_ids.shape[-1]
    if length + budget > POSITIONS:
        raise ValueError('Qwen prompt and reply exceeded the shared position budget')
    eos = model.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos)
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=budget, eos_token_id=eos,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    tokens = output[0, length:].tolist()
    return {'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'token_ids': tokens,
            'prompt_tokens': length, 'budget': budget, 'retained': len(history),
            'seconds': elapsed, 'eos': bool(tokens and tokens[-1] in stops)}


def compare_qwen35(folder, smoke):
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    begin = perf_counter()
    if transformers.__version__ != '5.3.0':
        raise ValueError('Run Qwen3.5 with the isolated Transformers 5.3.0 installation')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    names = (*QWEN35_HASHES, 'config.json', 'chat_template.jinja', 'tokenizer_config.json',
             'model.safetensors.index.json')
    sources = {name: digest(folder / name) for name in names}
    if any(sources[name] != expected for name, expected in QWEN35_HASHES.items()):
        raise ValueError('Qwen3.5 weights or tokenizer do not match the audited release')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        folder, local_files_only=True, dtype=torch.float16, device_map={'': 'cuda'}, attn_implementation='sdpa',
    ).eval().requires_grad_(False)
    model.generation_config.eos_token_id = sorted({
        tokenizer.eos_token_id, model.config.text_config.eos_token_id,
    })
    if not next(model.parameters()).is_cuda or model.config.text_config.model_type != 'qwen3_5_text':
        raise ValueError('Unexpected device or model type in Qwen3.5')
    if smoke:
        result = qwen35_answer(model, tokenizer, [], '用一句中文回答：把冰块放进温水，会发生什么？', 48)
        if not result['answer']:
            raise ValueError('Qwen3.5 returned an empty answer in the smoke test')
        print(result, flush=True)
        return

    def answer(history, question, budget):
        return qwen35_answer(model, tokenizer, history, question, budget)

    result = evaluate(answer)
    if any(digest(folder / name) != expected for name, expected in sources.items()):
        raise ValueError('A Qwen3.5 input file changed during evaluation')
    return {'native': result, 'sources': sources, 'revision': QWEN35_REVISION,
            'transformers': transformers.__version__, 'torch': str(torch.__version__),
            'dtype': str(next(model.parameters()).dtype), 'seconds': perf_counter() - begin,
            'peak_bytes': torch.cuda.max_memory_allocated()}


def compare_deepseek_pair(checkpoint):
    from scripts.audit_spora_open_chinese import compare_deepseek
    from scripts.audit_spora_dialogue_evidence import generate

    def comparisons(label, model, tokenizer):
        print(label, type(model).__name__, flush=True)

        def answer(history, question, budget):
            result = generate(model, tokenizer, history, question, POSITIONS, budget=budget)
            if result['retained'] != len(history) or result['budget'] != budget:
                raise ValueError('DeepSeek evaluation truncated the dialogue or generation budget')
            return result

        return evaluate(answer)

    return compare_deepseek(checkpoint, comparisons)


def main(args):
    if args.smoke and args.model != 'qwen35':
        raise ValueError('The smoke mode only loads Qwen3.5')
    if args.smoke:
        compare_qwen35(args.weights, smoke=True)
        return
    if args.output.exists() or not args.output.name.endswith('.json'):
        raise ValueError('Use a new JSON output file')
    manifest = Path(__file__).with_name('spora_fresh_dialogue_cases.py')
    sources = {path: digest(path) for path in (Path(__file__), manifest)}
    result = (compare_qwen35(args.weights, smoke=False) if args.model == 'qwen35'
              else compare_deepseek_pair(args.checkpoint))
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('The evaluation script or cases changed during execution')
    result.update(cases_sha256=sources[manifest], audit_sha256=sources[Path(__file__)],
                  early_budget=EARLY_BUDGET, final_budget=FINAL_BUDGET, positions=POSITIONS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('qwen35', 'deepseek'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, default=Path('D:/SPORA-runtime/models/Qwen3.5-0.8B-2fc06364'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/named-memory-layer9-packed-reply-v1/packed.pt'))
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
