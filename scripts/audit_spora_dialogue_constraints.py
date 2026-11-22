"""Separate context selection, instruction emphasis and response ranking in SPORA."""

import argparse
from collections import deque
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import _REPLY_LIMIT, _chat_inputs, ordinary_reply
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest


EMPHASIS = '请直接回答最后的问题。保留未撤销的限制，已更正的内容以最后一次为准。\n'
SCENES = {
    'revision': (
        ('社团的集合点原定钟楼。', '集合点是钟楼。', '地点改为码头。', '集合点改为码头。',
         '请只写最后确定的地点。', '码头', '钟楼'),
        ('木工作坊约在星期三。', '木工作坊约在星期三。', '改到星期五。', '木工作坊改到星期五。',
         '只写最新的星期几。', '星期五', '星期三'),
        ('短句只写「橘猫坐在椅子上」。', '橘猫坐在椅子上。', '把椅子换成窗台。', '橘猫坐在窗台上。',
         '只写改完后橘猫所在的位置。', '窗台', '椅子'),
        ('邮寄时收件人写罗芮。', '收件人是罗芮。', '收件人换成方冉。', '现在的收件人是方冉。',
         '请只写最新的收件人姓名。', '方冉', '罗芮'),
        ('参观路线先到竹园再去旧塔。', '先去竹园，再去旧塔。', '把顺序倒过来。', '先去旧塔，再去竹园。',
         '只写现在第一站的名称。', '旧塔', '竹园'),
        ('演出第一首歌是《晨光》。', '第一首歌是《晨光》。', '第一首改成《风铃》。', '第一首改成《风铃》。',
         '请只说最后决定的歌名，去掉书名号。', '风铃', '晨光'),
        ('我刚选了红色盒子。', '你选了红色盒子。', '改成蓝色的。', '现在选蓝色盒子。',
         '只写最后选的颜色。', '蓝色', '红色'),
        ('借用的工具是钳子。', '借用钳子。', '换成螺丝刀。', '借用螺丝刀。',
         '只回复现在要借的工具。', '螺丝刀', '钳子'),
    ),
    'earlier_constraint': (
        ('我只去免费的地方，社区展览免费，电影票二十元。', '你只考虑免费的地点。',
         '我今天下午有空。', '收到。', '社区展览和电影院，我应该选哪个？只写地点。', '社区展览', '电影院'),
        ('我今天不吃鸡蛋，白粥无蛋，蛋饼含蛋。', '今天需要选无蛋的。',
         '现在要吃早餐。', '明白。', '白粥和蛋饼选哪个？只写食物名。', '白粥', '蛋饼'),
        ('我不能爬楼梯，东口有电梯，西口只有楼梯。', '需要选能坐电梯的入口。',
         '我已经到院子了。', '知道了。', '东口和西口我该走哪个？只写入口名。', '东口', '西口'),
        ('我只收电子票，甲馆发纸票，乙馆发电子票。', '你只接受电子票。',
         '我计划下周参观。', '收到。', '甲馆和乙馆该选谁？只写馆名。', '乙馆', '甲馆'),
        ('我不喝甜的，柠檬水加糖，白水不加糖。', '你想喝不加糖的。',
         '我刚散步回来。', '好的。', '白水和柠檬水我选哪个？只写名称。', '白水', '柠檬水'),
        ('我只能带小盒，大盒放不进袋子。', '你只能带小盒。',
         '我准备出门了。', '好的。', '大盒和小盒该带哪个？只写盒名。', '小盒', '大盒'),
        ('我只取题名里的动词，「猫跳过围栏」的动作是跳。', '你想只提取动词。',
         '现在开始提取。', '明白。', '在「猫跳过围栏」里只回复动作。', '跳', '猫'),
        ('我只能在上午去，北馆上午开，南馆下午开。', '你只能上午去。',
         '今天有时间。', '收到。', '北馆和南馆我该去哪？只写馆名。', '北馆', '南馆'),
    ),
    'latest_constraint': (
        ('第一次想选烤饼。', '你先选了烤饼。', '现在只选汤面和烤饼里不油的，烤饼很油。', '收到。',
         '请只说应选的那一项。', '汤面', '烤饼'),
        ('旧路线走北门。', '旧路线走北门。', '北门维修封闭，南门能通行。', '收到。',
         '只写现在要走的门。', '南门', '北门'),
        ('原来通知去高台。', '原通知是高台。', '高台下雨太滑，现在只允许去室内。', '明白。',
         '高台和大厅里只选一个，写出名字。', '大厅', '高台'),
        ('我以前坐快车。', '你以前坐快车。', '快车这次要加价，慢车包含在票里，我只坐不加价的。', '收到。',
         '请只写现在要坐的车。', '慢车', '快车'),
        ('登记时我想拿黄色纸。', '你最初想要黄色纸。', '更新后只要白色纸，黄色纸作废。', '收到。',
         '只写当前需要的颜色。', '白色', '黄色'),
        ('我上次去了球场。', '你上次去了球场。', '这回球场闭馆，公园开着。', '明白。',
         '球场和公园选能进入的，只写地点。', '公园', '球场'),
        ('我之前买的是薄笔记本。', '你以前买薄笔记本。', '这次只能用厚本，薄本页数不够。', '收到。',
         '请只写现在要用的笔记本。', '厚本', '薄本'),
        ('旧方案使用上午场。', '旧方案是上午场。', '上午场已满，还有下午场。', '明白。',
         '现在可选哪个场次？只写场次名称。', '下午场', '上午场'),
    ),
}


@torch.inference_mode()
def likelihood(decoder, tokenizer, history, question, completion):
    cap = decoder.position_table_max_position or decoder.base.config.max_position_embeddings
    inputs, _ = _chat_inputs(tokenizer, history, question, cap - _REPLY_LIMIT)
    continuation = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    device = decoder.base.device
    target = torch.tensor([continuation], device=device)
    prompt = inputs['input_ids'].to(device)
    ids = torch.cat((prompt, target), dim=1)
    logits = decoder(input_ids=ids, use_cache=False, logits_to_keep=len(continuation) + 1).logits[:, :-1].float()
    scores = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    return float(scores.mean())


def evaluate(decoder, tokenizer):
    output = {}
    for family, cases in SCENES.items():
        rows = []
        for u1, a1, u2, a2, question, chosen, rejected in cases:
            history = deque(((u1, a1), (u2, a2)))
            variants = {}
            for label, used, prompt in (('full', history, question), ('recent', deque(((u2, a2),)), question),
                                         ('emphasis', history, EMPHASIS + question)):
                reply = ordinary_reply(decoder, tokenizer, used, prompt)
                normalized = reply['answer'].strip('。！!，,；;？?《》\n ')
                preferred = likelihood(decoder, tokenizer, used, prompt, chosen)
                foil = likelihood(decoder, tokenizer, used, prompt, rejected)
                variants[label] = {'reply': reply, 'strict': normalized == chosen,
                                   'preferred_ranked_above_foil': preferred > foil, 'ranking_margin': preferred - foil}
            rows.append({'history': list(history), 'question': question, 'preferred': chosen,
                         'foil': rejected, 'variants': variants})
        output[family] = rows
        print(family, {name: sum(row['variants'][name]['strict'] for row in rows) for name in variants}, flush=True)
    return output


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protected = {str(path): sha(path) for path in (args.checkpoint, args.database)}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed().eval(), MODEL) as runtime:
        baseline = frozen_digest(runtime.organism, runtime.state)
        evaluated = evaluate(runtime.decoder, tokenizer)
        if frozen_digest(runtime.organism, runtime.state) != baseline:
            raise ValueError('The language probe changed the saved individual')
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('The language probe changed a source file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'families': evaluated, 'emphasis': EMPHASIS, 'sources': protected,
                        'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
