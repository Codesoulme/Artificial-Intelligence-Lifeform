"""Run the fixed November 24 dialogue holdout in separate GPU processes."""

import argparse
import json
from pathlib import Path

from scripts.audit_local_named_chat import run
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated
from transformers import AutoModelForCausalLM, AutoTokenizer


CASES = (
    ('取件安排', (
        ('周颖上午去南站接人，下午到档案馆取件。请用一句话复述下午的安排。',
         '周颖下午(?:去|到)档案馆取件|下午(?:去|到)档案馆取件|下午在档案馆取件'),
        ('取件地点改到文化馆，时间照旧。请给周颖一句提醒。',
         '(?:周颖[，,])?下午(?:请)?(?:去|到)文化馆取件|请周颖下午(?:去|到)文化馆取件'),
        ('上午在哪接人来着？只回地点。', '南站'),
    )),
    ('标签调整', (
        ('旧纸箱标成待退，新纸箱标成留用。哪个要留着？只写纸箱名。', '新纸箱'),
        ('把两个标签对调。现在要退的是什么？只写纸箱名。', '新纸箱'),
    )),
    ('群消息', (
        ('请写一句通知发到群里，提醒大家彩排是下午三点，在二号厅。',
         '(?=.*下午(?:三|3)点)(?=.*(?:二|2)号厅)(?=.*彩排)[^。！？!?；;]+'),
        ('场地换成一号厅。把通知重新写一遍，时间也要写全。',
         '(?=.*下午(?:三|3)点)(?=.*(?:一|1)号厅)(?=.*彩排)(?!.*(?:二|2)号厅)[^。！？!?；;]+'),
    )),
    ('排队', (
        ('从头到尾，阿元站第一、阿沐站第二、阿南站第三。阿南前面紧挨着谁？只答称呼。', '阿沐'),
        ('阿元走到队尾，其他人相对位置照旧。阿南后面紧挨着谁？只答称呼。', '阿元'),
    )),
    ('汇合', (
        ('晴天在河边碰头，下雨就去书店。现在预报有雨，按约定只说地点。', '书店'),
        ('天气放晴了，但今天书店照常营业。按最初的约定只说碰头地点。', '河边'),
    )),
    ('报价', (
        ('一件材料原价二十元，运费三元。告诉我一共几元，只写阿拉伯数字。', '23'),
        ('材料改成十五元，运费照旧。现在总额多少？还是只写数字。', '18'),
    )),
    ('顺序与标点', (
        ('用顿号连起名单，先写何冬，再写徐晴，最后写任越。不要标题。', '何冬、徐晴、任越'),
        ('删掉最前面的人，把剩下的次序倒过来，标点沿用。', '任越、徐晴'),
    )),
    ('分工', (
        ('陶明负责订餐，叶青负责取票。要问取票的事找谁？只写名字。', '叶青'),
        ('叶青临时转去搬设备，取票交给孟西。现在取票找谁？只写名字。', '孟西'),
        ('还有谁管订餐？只写名字。', '陶明'),
    )),
    ('读书笔记', (
        ('第一条笔记标题叫潮汐，第二条叫松林。请单独发第二个标题。', '松林'),
        ('把第一条标题改成远山。现在按笔记顺序给我两个标题，中间用斜杠。', '远山/松林'),
    )),
    ('物资调拨', (
        ('甲仓有五把扳手，乙仓有两把。把甲仓的一把移到乙仓。乙仓现在有几把？只写数字。', '3'),
        ('再从乙仓移两把回甲仓，这时甲仓有几把？只写数字。', '6'),
    )),
    ('请假调整', (
        ('我打算周二请假，周三照常上班。请回一句说明周二的安排。',
         '周二(?:我)?(?:要|会|准备|计划|打算)?请假|(?:我)?(?:要|会|准备|计划|打算)?周二请假'),
        ('改成周三请假，两天安排对调。用一句话说明现在两天的安排。',
         '(?=.*(?:周|星期)二[^，,；;。]*上班)(?=.*(?:周|星期)三[^，,；;。]*请假)[^。！？!?]+'),
    )),
    ('转交物品', (
        ('姜原把水壶交给宋心，宋心再交给任荷。最后谁拿着水壶？只回名字。', '任荷'),
        ('任荷把水壶还给最初那个人。现在谁拿着？只回名字。', '姜原'),
    )),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = {
        'parent': Path('runs/named-memory-layer9-packed-reply-v1/packed.pt'),
        'candidate': Path('runs/named-memory-dialogue-midlayers-v1/integrated.pt'),
        'late': Path('runs/named-memory-dialogue-latelayers-v1/integrated.pt'),
        'layer12': Path('runs/named-memory-dialogue-layer12-v1/integrated.pt'),
        'layer19': Path('runs/named-memory-dialogue-layer19-v1/integrated.pt'),
    }
    parser.add_argument('condition', choices=('original', *checkpoints, 'qwen'))
    args = parser.parse_args()
    qwen = Path('D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798')
    output = Path('runs/named-memory-dialogue-midlayers-holdout-v1') / f'{args.condition}.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    instruct = args.condition == 'qwen'
    tokenizer = AutoTokenizer.from_pretrained(qwen if instruct else MODEL, local_files_only=True)
    source = checkpoints.get(args.condition)
    if instruct:
        source = qwen / 'model.safetensors'
    digest = sha(source) if source else None
    if instruct:
        model = AutoModelForCausalLM.from_pretrained(
            qwen, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation='sdpa',
        ).to('cuda').eval()
    else:
        model = load_full_streamed()
        if source:
            model, _, _ = load_integrated(source, model, MODEL, disposable_base=True)
    result = run(model, tokenizer, cases=CASES, instruct_model=instruct)
    if source and sha(source) != digest:
        raise ValueError('Evaluation source changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'sha256': digest, **result}, ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')
    print(args.condition, {key: value for key, value in result.items() if key != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
