"""Evaluate local chat on independent Chinese conversations with exact answers."""

import argparse
import json
from pathlib import Path
import re

from scripts.spora_named_memory_chat import LocalChat, MODEL, load_full_streamed, sha, torch
from spora.integrated_checkpoint import load_integrated
from transformers import AutoTokenizer


CASES = (
    ('数量', (('桌上原有六个杯子，拿走两个。只答剩下的数量。', '4|四'),
             ('现在放回一个，只答目前数量。', '5|五'))),
    ('更正', (('快递送往武汉，刚才说错了，改成长沙。只答最终目的地。', '长沙'),
             ('之前说错的目的地是哪座城？只答城名。', '武汉'))),
    ('对应', (('蓝包交给许宁，灰包交给周闻。只说灰包交给谁。', '周闻'),
             ('两个人交换手里的包。现在蓝包在谁手里？只答姓名。', '周闻'))),
    ('格式', (('把梨、桃用斜杠隔开，别加别的字。', '梨/桃'),
             ('互换顺序，保持同样的分隔符。', '桃/梨'))),
    ('顺序', (('工作安排先开窗，再铺桌布，最后摆盘。只回答第二项。', '铺桌布'),
             ('取消第二项以后，第一项后面做什么？只答那件事。', '摆盘'))),
    ('条件', (('如果刮大风就坐车，否则骑车。今天没刮大风。只答出行方式。', '骑车|骑自行车'),
             ('明天刮大风了，按刚才约定只答出行方式。', '坐车|乘车'))),
    ('指代', (('窗边那本是词典，门后那本是地图册。只说窗边是什么。', '词典'),
             ('再只说另一本是什么。', '地图册'))),
    ('约束', (('在牛肉、豆腐、猪肉里选一种素食材料。只答名字。', '豆腐'),
             ('又有玉米，把你选中的两种素食材料按先后用顿号隔开。', '豆腐、玉米'))),
    ('日期', (('讲座定在周四，后来推迟一天。只答最后是周几。', '周五|星期五'),
             ('改回原计划，只答最终是周几。', '周四|星期四'))),
    ('第三轮', (('第一个格子放笔，第二个格子放尺。只答第二个格子的物品。', '尺|尺子'),
               ('把第二个格子的物品换成书。只答换进去的物品。', '书'),
               ('第一个格子里还有什么？只答物品名称。', '笔'))),
)


INSTRUCTION = '用自然中文回答。先遵循这条消息的格式要求，再根据对话中明确给出的信息回答。'


class ModelRuntime:
    def __init__(self, model):
        self.model = model

    def generate_ordinary(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


@torch.inference_mode()
def reasoning_answer(model, tokenizer, history, question, seed, budget):
    messages = [*history, {'role': 'user', 'content': question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt + '\n', return_tensors='pt', add_special_tokens=False).to('cuda')
    torch.manual_seed(seed)
    ids = model.generate(**inputs, max_new_tokens=budget, use_cache=True, do_sample=True,
                         temperature=.6, top_p=.95, pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    tail = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    marker = '</think>'
    if marker not in tail:
        return '', False
    answer = tail.partition(marker)[2]
    if tokenizer.eos_token is not None:
        answer = answer.partition(tokenizer.eos_token)[0]
    return answer.strip(), True


def run(model, tokenizer, instruction=None, *, reasoning_budget=0, history_mode='live'):
    outcomes = []
    for case_index, (group, turns) in enumerate(CASES):
        chat = LocalChat(ModelRuntime(model), tokenizer, None, None)
        rows, history = [], []
        for turn_index, (question, expected) in enumerate(turns):
            prompt = question if instruction is None else instruction + '\n' + question
            finished = None
            if history_mode == 'users_only':
                earlier = '\n'.join(text for text, _ in turns[:turn_index])
                if earlier:
                    prompt = '我先前说过：\n' + earlier + '\n我这次的问题：\n' + prompt
                chat.turns = []
            elif history_mode == 'reference':
                chat.turns = [
                    {'role': role, 'content': content}
                    for previous, reference in turns[:turn_index]
                    for role, content in (('user', previous), ('assistant', reference.split('|')[0]))
                ]
            if reasoning_budget:
                answer, finished = reasoning_answer(model, tokenizer, history, prompt,
                                                     3832 + case_index * 20 + turn_index, reasoning_budget)
                history.extend(({'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}))
            else:
                answer = chat.talk(prompt)
            normalized = re.sub(r'\s+', '', answer).strip('。.!！').replace('／', '/')
            rows.append({'question': question, 'expected': expected, 'answer': answer,
                         'correct': re.fullmatch(expected, normalized) is not None,
                         'reasoning_finished': finished})
        outcomes.append({'case': group, 'turns': rows, 'all_turns_correct': all(row['correct'] for row in rows)})
    return {'cases_passed': sum(row['all_turns_correct'] for row in outcomes), 'cases': len(outcomes),
            'turns_passed': sum(turn['correct'] for row in outcomes for turn in row['turns']),
            'turns': sum(len(row['turns']) for row in outcomes), 'rows': outcomes}


def main(checkpoint, output, reasoning_budget, compare_history):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    digest = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    if compare_history:
        decoder, _, _ = load_integrated(checkpoint, base, MODEL)
        results = {mode: run(decoder, tokenizer, history_mode=mode) for mode in ('live', 'reference', 'users_only')}
    elif reasoning_budget:
        decoder, _, _ = load_integrated(checkpoint, base, MODEL)
        results = {'spora_open_reasoning': run(decoder, tokenizer, reasoning_budget=reasoning_budget)}
    else:
        original = run(base, tokenizer)
        decoder, _, _ = load_integrated(checkpoint, base, MODEL)
        results = {'original': original, 'spora': run(decoder, tokenizer),
                   'spora_with_instruction': run(decoder, tokenizer, INSTRUCTION)}
    for label in results:
        print(label, {key: value for key, value in results[label].items() if key != 'rows'}, flush=True)
    if sha(checkpoint) != digest:
        raise ValueError('Checkpoint changed during the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256': digest, 'instruction': INSTRUCTION,
                   'reasoning_budget': reasoning_budget, 'results': results},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reasoning-budget', type=int, default=0)
    parser.add_argument('--compare-history', action='store_true')
    args = parser.parse_args()
    if args.reasoning_budget < 0 or (args.compare_history and args.reasoning_budget):
        parser.error('Choose either a positive reasoning budget or the history comparison')
    main(**vars(args))
