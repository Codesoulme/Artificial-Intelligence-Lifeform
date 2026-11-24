"""Evaluate a saved dialogue model on composition and open development prompts."""

import argparse
import json
from pathlib import Path
import re

from scripts.audit_local_named_chat import run
from scripts.audit_named_dialogue_midlayers_holdout import CASES
from scripts.audit_spora_chat_backbones import DEVELOPMENT, evaluate, reply
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_composition import DATA, score_rollouts, settings_for
from spora.integrated_checkpoint import load_integrated
from transformers import AutoTokenizer


HISTORY_PAIRS = (
    ('预约地点', '桐叶馆', '月泉亭', '合照本来在{old}，今天临时通知改在{new}。', '最后在哪里拍照？只写地点。', 'new'),
    ('任务交接', '石芸', '余闻', '开始请{old}校对，后来由{new}接手了校对。', '现在校对该找谁？只写姓名。', 'new'),
    ('取件对象', '相册', '日历', '现在保留{old}，打包{new}带走。', '要带走什么？只写物品名。', 'new'),
    ('餐单撤项', '莲藕', '芋头', '餐单上有{old}和{new}，{old}今天停售。', '店里还能提供哪道菜？只写菜名。', 'new'),
    ('交货方式', '自提', '送货', '提货方式从{old}变更为{new}。', '单据目前写的是什么方式？只写方式。', 'new'),
    ('编号回溯', '五号', '八号', '我先抽到{old}，后来换到{new}。', '我一开始抽到了几号？只写编号。', 'old'),
    ('约定顺序', '擦桌', '收椅', '我们先做{old}，第二步是{new}。', '第一步做什么？只写动作。', 'old'),
    ('保存来源', '云桥集', '石巷录', '标题抄自{old}，其他内容抄自{new}。', '标题来自哪里？只写书名。', 'old'),
)


def score_history_pairs(decoder, tokenizer, settings):
    rows = []
    for name, left, right, first, question, target in HISTORY_PAIRS:
        turns = []
        for old, new in ((left, right), (right, left)):
            stale = f'你说的是{left}，我记下了。'
            history = [{'role': 'user', 'content': first.format(old=old, new=new)},
                       {'role': 'assistant', 'content': stale}]
            expected = new if target == 'new' else old
            generated = reply(decoder, tokenizer, history, question, **settings)
            answer = re.sub(r'\s+', '', generated['answer']).strip('。.!！《》')
            turns.append({'history': history, 'question': question, 'expected': expected,
                          **generated, 'correct': answer == expected})
        rows.append({'case': name, 'turns': turns, 'passed': all(turn['correct'] for turn in turns)})
    print('History pairs', sum(row['passed'] for row in rows), '/', len(rows), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--short-holdout', action='store_true')
    parser.add_argument('--history-pairs', choices=('include', 'only'))
    parser.add_argument('--development-only', type=Path,
                        help='Evaluate just the supplied open-dialogue development scenarios')
    parser.add_argument('--stage-output-projection', action='store_true',
                        help='Keep the output projection on CPU until checkpoint assembly finishes')
    args = parser.parse_args()
    if args.development_only is not None and (args.short_holdout or args.history_pairs):
        parser.error('--development-only runs without the short or paired diagnostics')
    if args.output.exists():
        raise FileExistsError(args.output)
    development_path = args.development_only or DEVELOPMENT
    inputs = (development_path,) if args.development_only is not None else (DATA, DEVELOPMENT)
    sources = {str(path): sha(path) for path in (args.checkpoint, *inputs)}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    print('Loading verified base weights', flush=True)
    base = load_full_streamed(stage_output_projection=args.stage_output_projection)
    print('Assembling checkpoint', flush=True)
    decoder, _, _ = load_integrated(args.checkpoint, base, MODEL, disposable_base=True)
    if args.stage_output_projection:
        torch.cuda.empty_cache()
        print('Moving output projection to CUDA', flush=True)
        base.lm_head.to(base.device)
    settings = settings_for(decoder, tokenizer, 96)
    result = {'sources': sources}
    if args.history_pairs:
        result['history_pairs'] = score_history_pairs(decoder, tokenizer, settings)
    if args.short_holdout:
        result['short'] = run(decoder, tokenizer, cases=CASES)
        print('Short holdout', {key: value for key, value in result['short'].items() if key != 'rows'}, flush=True)
    if args.history_pairs != 'only':
        if args.development_only is None:
            composition = json.loads(DATA.read_text(encoding='utf-8'))['evaluation']
            result['composition'] = score_rollouts(decoder, tokenizer, composition, settings=settings)
        development = json.loads(development_path.read_text(encoding='utf-8'))['scenarios']
        result['open'], _ = evaluate(decoder, tokenizer, development, {}, settings)
    if any(sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('Evaluation input changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
