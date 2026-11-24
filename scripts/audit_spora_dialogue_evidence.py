"""Compare identical Chinese task evidence across single- and multi-turn inputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts.deepseek_continuous_memory_chat import load_chat, ordinary_context_ids
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, AutoTokenizer, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json


BUDGET = 48
POSITIONS = 256


def turn(text, required=(), forbidden=(), *, length=None, start=None, end=None, exact=None):
    if length is not None:
        text = f'{text}最多{length}字。'
    return {'text': text, 'required': tuple(required), 'forbidden': tuple(forbidden),
            'length': length, 'start': start, 'end': end, 'exact': exact}


CASES = (
    ('集合通知', (
        turn('写一句集合通知，地点是西门，请穿蓝色马甲。', ('西门', '蓝色马甲')),
        turn('地点改成东门，衣服要求先保留。', ('东门', '蓝色马甲'), ('西门',)),
        turn('衣服要求取消，只提醒带水。写成一句，别出现旧地点。', ('东门', '带水'), ('西门', '马甲'), length=32),
    ), '写一句集合通知。在东门集合，提醒带水，省略衣服要求。整句最多32字。'),
    ('周末展馆', (
        turn('写一句参观邀约。周六上午九点去湖边展馆。', ('周六', '九点', '湖边展馆')),
        turn('改到周日，地点、晨间时间照旧。', ('周日', '九点', '湖边展馆'), ('周六',)),
        turn('把九点改成十点，放进同一句，保留最新日期和地点。', ('周日', '十点', '湖边展馆'), ('周六', '九点'), length=45),
    ), '用一句不超过45字的话邀约参观。周日上午十点在湖边展馆。'),
    ('解释露水', (
        turn('用一句话给孩子解释露水，要出现“水汽”。', ('水汽',)),
        turn('继续保留水汽，补上变成小水滴的过程。', ('水汽', '小水滴')),
        turn('把刚才的说法缩短成一句。保留两个词，也别写“冷凝”这个术语。', ('水汽', '小水滴'), ('冷凝',), length=32),
    ), '用一句不超过32字的话给孩子解释露水，须用“水汽”和“小水滴”，避开“冷凝”。'),
    ('午餐留言', (
        turn('给同事留一句午餐留言，请他带上饭盒和苹果。', ('饭盒', '苹果')),
        turn('把苹果换成梨，饭盒依旧要带。', ('饭盒', '梨'), ('苹果',)),
        turn('留言末尾加“谢谢”，不要恢复已经换掉的水果。', ('饭盒', '梨'), ('苹果',), length=40, end='谢谢'),
    ), '给同事留一句不超过40字的午餐留言，请他带上饭盒和梨。末尾写“谢谢”。'),
    ('早餐准备', (
        turn('写一句早餐安排。准备豆浆和鸡蛋。', ('豆浆', '鸡蛋')),
        turn('改成无蛋餐，豆浆保留，用馒头替换鸡蛋。', ('豆浆', '馒头'), ('鸡蛋',)),
        turn('再加橙子，简短列出成品，避开刚取消的食物。', ('豆浆', '馒头', '橙子'), ('鸡蛋',), length=30),
    ), '简短列出无蛋早餐的成品，三样是豆浆、馒头和橙子，最多30字。'),
    ('借书安排', (
        turn('写条提醒：宁岚借的是《云上信》，星期四归还。', ('宁岚', '云上信', '星期四')),
        turn('书名记错了，换成《远山图》，归还时间照旧。', ('宁岚', '远山图', '星期四'), ('云上信',)),
        turn('归还时间再推到星期五，保留正确人名和书名，只用一句话。',
             ('宁岚', '远山图', '星期五'), ('云上信', '星期四'), length=44),
    ), '只用一句不超过44字的话提醒宁岚星期五归还《远山图》。'),
    ('等待确认', (
        turn('我在准备发活动通知，还得等老师确认。先回复“等确认”。', exact='等确认'),
        turn('老师还没有回复。继续等待，只回复“仍在等”。', exact='仍在等'),
        turn('老师已经确认。只输出“可以发布”。', exact='可以发布'),
    ), '老师先前尚未确认，现已确认活动通知。只输出“可以发布”。'),
    ('背包选择', (
        turn('在松针包和溪石包里选一个。松针包贵一些但较轻，先推荐轻的。', ('松针包',)),
        turn('现在只看价格，溪石包更便宜。再选一次。', ('溪石包',)),
        turn('沿用最新标准，只报包名，不给理由。', exact='溪石包'),
    ), '候选是松针包和溪石包。松针包轻但贵，溪石包便宜。按价格选便宜的，只报包名。'),
    ('交通选择', (
        turn('两趟车叫山雀号和银杉号。山雀号九点发车，银杉号十点发车。先报较早的车名。', ('山雀号',)),
        turn('我九点半才到车站。请改选赶得上的车。', ('银杉号',)),
        turn('还是同一个出发条件。最终只写车名。', exact='银杉号'),
    ), '山雀号九点发车，银杉号十点发车。我九点半才到车站。只写能赶上的车名。'),
    ('提示语格式', (
        turn('拟一句图书馆标语，必须有“安静”两个字。', ('安静',)),
        turn('把“安静”换成“轻声”，不要出现原来的词。', ('轻声',), ('安静',)),
        turn('保持新词，标语开头用“提示”，末尾写“谢谢”。', ('提示', '轻声'), ('安静',), length=28, start='提示', end='谢谢'),
    ), '拟一句图书馆标语，开头用“提示”，使用“轻声”，避开“安静”，末尾写“谢谢”。最多28字。'),
    ('整理事项', (
        turn('请把三项工作都写出：拍照、归档、寄送。', ('拍照', '归档', '寄送')),
        turn('取消拍照，增加盖章，其他照旧。', ('归档', '寄送', '盖章'), ('拍照',)),
        turn('寄送也取消了。只用顿号连接当前的两个事项。', exact='归档、盖章'),
    ), '当前只剩归档和盖章，按这个顺序用顿号连接两个事项，直接给出结果。'),
    ('门口提醒', (
        turn('用礼貌语气写一句提醒，周三前在北门交表。', ('周三', '北门', '交表')),
        turn('日期不变，把北门换成服务台。', ('周三', '服务台', '交表'), ('北门',)),
        turn('把这句改成问句，别丢日期和交表地点。', ('周三', '服务台', '交表'), ('北门',), length=36, end='？'),
    ), '用不超过36字的礼貌问句确认对方能否在周三前到服务台交表，末尾用问号。'),
)

ASSISTANT_CONTROLS = (
    ('我准备了三个命名方案：纸灯、竹影、远窗。', turn('你刚才列的第二个名字是什么？只写那个名字。', exact='竹影')),
    ('两处地点备选是银杏亭和晨风台。推荐较安静的晨风台。', turn('只告诉我你刚才推荐的地点。', exact='晨风台')),
    ('本轮采用的确认口令是“青鸟到站”。', turn('只把你刚才定的确认口令重复一遍。', exact='青鸟到站')),
    ('我把便签暂命名为“木槿”，副标题是“周边提醒”。', turn('副标题先别说，只写你刚定的名字。', exact='木槿')),
)

DEVELOPMENT_TARGETS = {
    '集合通知': '请在东门集合，记得带水。',
    '周末展馆': '周日上午十点，一起去湖边展馆参观吧。',
    '解释露水': '空气里的水汽遇冷会变成小水滴，这就是露水。',
    '午餐留言': '午餐请带上饭盒和梨，谢谢。',
    '早餐准备': '早餐准备豆浆、馒头和橙子。',
    '借书安排': '宁岚，记得星期五归还《远山图》。',
    '等待确认': '可以发布',
    '背包选择': '溪石包',
    '交通选择': '银杉号',
    '提示语格式': '提示：进入图书馆请轻声，谢谢。',
    '整理事项': '归档、盖章',
    '门口提醒': '请问你方便在周三前到服务台交表吗？',
}


def scores(text, spec):
    normalized = text.strip(' \n\t。！!？?，,「」“”')
    ending = spec['end']
    ended_correctly = text.rstrip().endswith(('？', '?')) if ending == '？' else ending is None or normalized.endswith(ending)
    checks = {'nonempty': bool(normalized), 'required': all(word in text for word in spec['required']),
              'forbidden': not any(word in text for word in spec['forbidden']),
              'length': spec['length'] is None or len(text) <= spec['length'],
              'start': spec['start'] is None or normalized.startswith(spec['start']),
              'ending': ended_correctly, 'exact': spec['exact'] is None or normalized == spec['exact']}
    return {**checks, 'pass': all(checks.values())}


@torch.inference_mode()
def generate(model, tokenizer, history, question, positions, *, budget=BUDGET):
    current = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                            add_generation_prompt=True)
    current_ids = tokenizer.encode(current + '</think>\n\n', add_special_tokens=False)
    prompt, retained = ordinary_context_ids(tokenizer, history, question, positions - budget, current_ids)
    available = min(budget, positions - len(prompt))
    if available < 1:
        raise ValueError(f'Task exceeded the decoder position limit: {question}')
    ids = torch.tensor([prompt], device='cuda')
    began = perf_counter()
    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                               max_new_tokens=available, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)[0, len(prompt):].tolist()
    return {'answer': tokenizer.decode(generated, skip_special_tokens=True).strip(), 'token_ids': generated,
            'prompt_tokens': len(prompt), 'retained': retained, 'budget': available,
            'eos': bool(generated and generated[-1] == tokenizer.eos_token_id), 'seconds': perf_counter() - began}


def evaluate(model, tokenizer, positions):
    counts = Counter()
    rows = []
    for name, stages, resolved in CASES:
        raw_history, raw = [], []
        for index, spec in enumerate(stages):
            result = generate(model, tokenizer, raw_history, spec['text'], positions)
            result['score'] = scores(result['answer'], spec)
            counts[f'raw_turn_{index+1}'] += result['score']['pass']
            raw.append(result)
            raw_history.append((spec['text'], result['answer']))
        final = stages[-1]
        conditions = {'raw': raw[-1], 'full': generate(model, tokenizer, [], '\n'.join(t['text'] for t in stages), positions),
                      'neutral': generate(model, tokenizer, [(t['text'], '收到。') for t in stages[:-1]], final['text'], positions),
                      'resolved': generate(model, tokenizer, [], resolved, positions)}
        for mode, result in conditions.items():
            result['score'] = scores(result['answer'], final)
            counts[f'{mode}_pass'] += result['score']['pass']
            counts[f'{mode}_eos'] += result['eos']
        passed = {mode: result['score']['pass'] for mode, result in conditions.items()}
        counts.update({'full_success_raw_failure': passed['full'] and not passed['raw'],
                       'neutral_recovers_raw': passed['neutral'] and not passed['raw'],
                       'resolved_recovers_raw': passed['resolved'] and not passed['raw'],
                       'raw_retained_every_user_turn': raw[-1]['retained'] == len(stages) - 1})
        rows.append({'case': name, 'stages': stages, 'resolved_prompt': resolved, 'raw_turns': raw, 'final': conditions})
        print(name, passed, flush=True)
    controls = []
    for assistant, spec in ASSISTANT_CONTROLS:
        original = '请先给我这项任务的准备内容。'
        values = {mode: generate(model, tokenizer, [(original, previous)], spec['text'], positions)
                  for mode, previous in (('with_assistant', assistant), ('neutral', '收到。'))}
        for mode, result in values.items():
            result['score'] = scores(result['answer'], spec)
            counts[f'references_{mode}'] += result['score']['pass']
        controls.append({'assistant': assistant, 'target': spec, 'outputs': values})
    return {'summary': dict(counts), 'rows': rows, 'assistant_reference_controls': controls}


def load_spora_for_eval(path):
    working = path / 'unwritten-individual.sqlite3'
    if working.exists():
        raise FileExistsError(working)
    return load_chat(working, use_inherited_router=True, use_retrained_gate=True)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    path = args.output.resolve()
    path.mkdir(parents=True, exist_ok=True)
    destination = path / f'{args.model}.json'
    if destination.exists():
        raise FileExistsError(destination)
    source = MODEL / 'model.safetensors'
    positions = POSITIONS
    if args.model == 'spora':
        # The same decoder used by the Windows chat entry, with a new disposable memory path.
        chat = load_spora_for_eval(path)
        model, tokenizer = chat.decoder, chat.tokenizer
        positions = min(positions, chat.position_limit)
        runtime = chat.awake_individual
    else:
        model = load_full_streamed()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        runtime = {}
    result = evaluate(model, tokenizer, positions)
    if sha(source) != WEIGHT_SHA:
        raise ValueError('The original model weights changed')
    store_json(destination, {**result, 'model': args.model, 'runtime': runtime, 'base_sha256': WEIGHT_SHA,
                             'position_limit': positions, 'reply_budget': BUDGET, 'seconds': perf_counter() - started,
                             'peak_bytes': torch.cuda.max_memory_allocated()})
    print(args.model, json.dumps(result['summary']), flush=True)


def self_resolve(output):
    """Test whether the same decoder can recover the current user requirements."""
    started = perf_counter()
    path = output.resolve()
    destination = path / 'self-resolved.json'
    if destination.exists():
        raise FileExistsError(destination)
    reference = path / 'spora.json'
    prior_digest = sha(reference)
    prior = json.loads(reference.read_text(encoding='utf-8'))
    chat = load_spora_for_eval(path)
    if chat.awake_individual != prior['runtime']:
        raise ValueError('The SPORA checkpoint differs from the dialogue comparison')
    positions = prior['position_limit']
    counts = Counter()
    rows = []
    for row in prior['rows']:
        stages = row['stages']
        evidence = '\n'.join(f'{index}. {stage["text"]}' for index, stage in enumerate(stages, 1))
        prompt = ('按顺序阅读用户的要求。用户明确改过的内容用新要求，未改的要求保留。'
                  '只写一条合并后的任务指令。不要完成任务，也不要复述已取消的条件。\n' + evidence)
        draft = generate(chat.decoder, chat.tokenizer, [], prompt, positions, budget=64)
        generated = generate(chat.decoder, chat.tokenizer, [],
                             f'请直接完成下面的用户任务。\n{draft["answer"]}', positions)
        final = stages[-1]
        generated['score'] = scores(generated['answer'], final)
        required_present = all(word in draft['answer'] for word in final['required'])
        stale_absent = not any(word in draft['answer'] for word in final['forbidden'])
        counts.update({'total': 1, 'draft_required_present': required_present, 'draft_stale_absent': stale_absent,
                       'draft_eos': draft['eos'],
                       'output_pass': generated['score']['pass'], 'output_eos': generated['eos'],
                       'recovers_raw': generated['score']['pass'] and not row['final']['raw']['score']['pass']})
        print(row['case'], generated['score']['pass'], flush=True)
        rows.append({'case': row['case'], 'rewrite_prompt': prompt, 'draft': draft, 'answer': generated,
                     'draft_required_present': required_present, 'draft_stale_absent': stale_absent})
    if sha(reference) != prior_digest:
        raise ValueError('The paired comparison changed during replay')
    store_json(destination, {'source_sha256': prior_digest, 'summary': dict(counts), 'rows': rows,
                             'seconds': perf_counter() - started})
    print(json.dumps(counts), flush=True)


def prepare_review(output):
    """Save candidate supervision from clean successes for semantic review."""
    sources = {name: output / f'{name}.json' for name in ('spora', 'base')}
    groups, observations = {}, {}
    for name, path in sources.items():
        result = json.loads(path.read_text(encoding='utf-8'))
        examples = []
        clean_successes = 0
        for row in result['rows']:
            final = row['final']
            if final['raw']['score']['pass']:
                continue
            spec = row['stages'][-1]
            gold = DEVELOPMENT_TARGETS[row['case']]
            if not scores(gold, spec)['pass']:
                raise ValueError(f'Handwritten target violates the task: {row["case"]}')
            clean = {source: final[source]['answer'] for source in ('full', 'resolved')
                     if final[source]['score']['pass'] and final[source]['eos']}
            clean_successes += bool(clean)
            previous = [{role: text for role, text in (('user', turn['text']), ('assistant', reply['answer']))}
                        for turn, reply in zip(row['stages'][:-1], row['raw_turns'][:-1], strict=True)]
            examples.append({'case': row['case'], 'history': previous, 'request': spec['text'],
                             'raw_answer': final['raw']['answer'], 'answer_reference': gold,
                             'clean_rule_matches_for_review': clean, 'criteria': spec})
        groups[name] = examples
        observations[name] = clean_successes
    report = {'use': 'development only; reserve independently authored cases for evaluation',
              'sources': {name: sha(path) for name, path in sources.items()}, 'clean_rule_matches': observations,
              'counts': {name: len(rows) for name, rows in groups.items()}, 'candidates': groups}
    store_json(output / 'recovery-review.json', report)
    print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', choices=('spora', 'base'))
    mode.add_argument('--prepare-review', action='store_true')
    mode.add_argument('--self-resolve', action='store_true')
    args = parser.parse_args()
    if args.prepare_review:
        prepare_review(args.output)
    elif args.self_resolve:
        self_resolve(args.output)
    else:
        main(args)
