"""Test whether short finished answers help DeepSeek SPORA deliver usable replies."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_dialogue_evidence import generate
from scripts.audit_spora_fresh_dialogue import FINAL_BUDGET, POSITIONS, digest
from scripts.audit_spora_open_chinese import compare_deepseek
from scripts.spora_fresh_dialogue_cases import CASES
from scripts.train_spora_recurrent_counterfactual import store_json


GUIDANCE = '直接回答下面的问题，给出请求的成品。保留所有现行要求。信息不足时明确说明。\n\n'
DEMOS = (
    ('直接给同事发一句话，请他周二下班后到西桥取一枚印章。',
     '你好，请周二下班后到西桥取一枚印章，谢谢！'),
    ('票务页面只写“申请已提交”，能确定出票了吗？只给一个确认方法。',
     '现在还无法确定是否出票，可以在购票平台查看订单详情。'),
    ('用一句可以直接发给合唱团的通知，说明今晚七点从音乐厅门口入场，并自带谱夹。',
     '请大家今晚七点从音乐厅门口入场，记得带上自己的谱夹。'),
)
NEUTRAL = tuple((question, '好的，已收到。') for question, _ in DEMOS)


def evaluate(model, tokenizer, previous):
    rows = []
    for case, prior in zip(CASES, previous['integrated']['rows'], strict=True):
        if prior['name'] != case.name or prior['resolved_prompt'] != case.resolved_prompt:
            raise ValueError('The saved answer does not match the fixed dialogue manifest')
        question = case.resolved_prompt
        outputs = {}
        for label, history, prompt in (
            ('plain', (), question),
            ('guidance', (), GUIDANCE + question),
            ('neutral', NEUTRAL, GUIDANCE + question),
            ('demonstrations', DEMOS, GUIDANCE + question),
        ):
            output = generate(model, tokenizer, history, prompt, POSITIONS, budget=FINAL_BUDGET)
            if output['retained'] != len(history) or output['budget'] != FINAL_BUDGET:
                raise ValueError('A prompt condition was truncated')
            outputs[label] = output
        if outputs['plain']['token_ids'] != prior['resolved']['token_ids']:
            raise ValueError('The frozen unmodified input did not reproduce its saved answer')
        rows.append({'name': case.name, 'question': question, 'outputs': outputs})
        print(case.name, {label: output['eos'] for label, output in outputs.items()}, flush=True)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = (args.reference, Path(__file__), Path(__file__).with_name('spora_fresh_dialogue_cases.py'))
    sources = {path: digest(path) for path in inputs}
    previous = json.loads(args.reference.read_text(encoding='utf-8'))
    if previous['cases_sha256'] != sources[inputs[-1]]:
        raise ValueError('Expected reference outputs from the fixed dialogue manifest')

    def comparisons(name, model, tokenizer):
        return evaluate(model, tokenizer, previous) if name == 'integrated' else None

    result = compare_deepseek(args.checkpoint, comparisons)
    if (result['individual_sha256'] != previous['individual_sha256']
            or result['sources'] != previous['sources']):
        raise ValueError('The frozen base or individual differs from the earlier run')
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('An evaluation source changed during inference')
    result.pop('native')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store_json(args.output, {**result, 'evaluation_sha256': {str(path): value for path, value in sources.items()},
                            'guidance': GUIDANCE, 'demonstrations': DEMOS, 'neutral': NEUTRAL})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/named-memory-layer9-packed-reply-v1/packed.pt'))
    main(parser.parse_args())
