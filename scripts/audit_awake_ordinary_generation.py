"""Measure answer quality, completion and latency across DeepSeek decoding settings."""

import argparse
import time
from pathlib import Path

from deepseek_awake_recall_intent import DEFAULT_DATABASE
from deepseek_continuous_memory_chat import load_chat
from deepseek_output_aligned_memory import sha, torch, write


QUESTIONS = (
    '为什么热水会慢慢冷却？',
    '推荐一杯少糖的饮料。',
    '三只杯子各装两块冰，一共多少块？',
    '请把「她最喜欢喝什么」翻译成英语。',
    '如果明天降温，出门怎么穿比较好？',
    '晚上想放松一下，给我两个简单建议。',
)
SETTINGS = (
    ('closed_64', False, 64, False),
    ('closed_192', False, 192, False),
    ('open_256', True, 256, True),
)


@torch.inference_mode()
def main(database, output):
    digest = sha(database)
    chat = load_chat(database)
    results = {}
    for name, open_think, limit, sample in SETTINGS:
        rows = []
        for index, question in enumerate(QUESTIONS):
            text = chat.tokenizer.apply_chat_template([dict(role='user', content=question)], tokenize=False,
                                                      add_generation_prompt=True)
            if not text.rstrip().endswith('<think>'):
                raise ValueError('Unexpected generation template')
            text += '\n' if open_think else '</think>\n\n'
            encoded = chat.tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
            torch.manual_seed(82511 + index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = chat.decoder.generate(**encoded, do_sample=sample, max_new_tokens=limit,
                                              temperature=.6 if sample else None, top_p=.95 if sample else None,
                                              pad_token_id=chat.tokenizer.pad_token_id,
                                              eos_token_id=chat.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = generated[0, encoded['input_ids'].shape[1]:].tolist()
            raw = chat.tokenizer.decode(tokens, skip_special_tokens=True)
            answer = raw.partition('</think>')[2] if open_think else raw
            row = dict(question=question, limit=limit, tokens=len(tokens), seconds=round(elapsed, 3),
                       eos=bool(tokens and tokens[-1] == chat.tokenizer.eos_token_id), answer=answer.strip())
            rows.append(row)
            print(name, row, flush=True)
        results[name] = rows
    if sha(database) != digest:
        raise ValueError('Source memory database changed during evaluation')
    write(output, dict(results=results, database_sha256=digest, decoder=chat.recall_fingerprint))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    main(args.database, args.output)
