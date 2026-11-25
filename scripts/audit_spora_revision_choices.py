"""Test whether frozen SPORA decoders distinguish current and withdrawn instructions."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import POSITIONS, load_spora_for_eval, store_json
from scripts.audit_named_memory_qwen_structure_control import CHECKPOINT_SHA, QWEN_SHA, attach_control
from scripts.deepseek_local_baseline import AutoTokenizer, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from spora.integrated_checkpoint import _digest
from spora.language_training import frozen_digest


# Each pair keeps its first two user turns and both candidates fixed. Changing
# only the last turn reverses which of the two candidates is valid.
PAIRS = (
    ('药箱清单', ('写便签说带纱布和胶带。', '其余照旧，另加消毒棉。'),
     ('起初提到的第一件免了，别写进便签。', '起初提到的第二件免了，别写进便签。'),
     ('带胶带和消毒棉。', '带纱布和消毒棉。')),
    ('图册标题', ('标题先用《夜航图》，副标题用《石径》。', '备选标题《潮汐》，副标题继续保留。'),
     ('备选只是拿来比较，封面还是用原定标题。', '封面采用备选标题，原定标题撤掉。'),
     ('标题《夜航图》，副标题《石径》。', '标题《潮汐》，副标题《石径》。')),
    ('维修通知', ('写维修通知，周三开工。用“暂停使用”四个字。', '改成周四开工，其他照旧。'),
     ('最后用较早提到的日期，措辞按现有的。', '最后用较晚提到的日期，措辞按现有的。'),
     ('周三开工，暂停使用。', '周四开工，暂停使用。')),
    ('柜台方向', ('寄存处在南厅，取件记得带凭条。', '北厅还在测试，先不要替换正式寄存处。'),
     ('在通知里只引用“启用北厅”作为反例。实际地点照旧。',
      '在测试结束后的通知里正式写“启用北厅”。取件规则照旧。'),
     ('取件到南厅，带凭条。', '取件到北厅，带凭条。')),
    ('短答通知', ('演练结束只报“灯灭”，现阶段只报“灯亮”。', '排练还在进行，结束口令提前准备。'),
     ('主持人说仍在排练，只报当下该报的口令。', '主持人宣布排练结束，只报当下该报的口令。'),
     ('灯亮', '灯灭')),
    ('抄送名单', ('发给闻雪和若松，另抄送白鹭。', '若松调离，换成安桥，其他人保留。'),
     ('刚说的人员变动算了，用最初的名单。', '刚说的人员变动确认，其余照旧。'),
     ('发给闻雪和若松，抄送白鹭。', '发给闻雪和安桥，抄送白鹭。')),
    ('校刊用词', ('校刊标题最初要求含“春潮”，开头加“新刊”。', '把指定词换成“青苔”。开头保留。'),
     ('主编恢复最初指定词，沿用当前开头。', '主编认可刚换的指定词，沿用当前开头。'),
     ('开头用“新刊”，指定词是“春潮”。', '开头用“新刊”，指定词是“青苔”。')),
    ('借还安排', ('约费宁去竹台还器材，时间是午后。', '涧桥也开门，但那只是另一个可选点。'),
     ('涧桥是备选，还是按正式地点办。', '正式地点就用那个备选点，时间与人照旧。'),
     ('费宁午后去竹台还器材。', '费宁午后去涧桥还器材。')),
)


def prompt_for(tokenizer, transcript, question, deepseek):
    history = '\n'.join(f'第{index}轮：{turn}' for index, turn in transcript)
    message = f'以下为按时间顺序记录的用户发言：\n{history}\n\n{question}'
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': message}],
                                           tokenize=False, add_generation_prompt=True)
    if deepseek:
        prompt += '</think>\n\n'
    return prompt


@torch.inference_mode()
def logit_margin(model, tokenizer, transcript, choices, labels, deepseek):
    question = ('哪项符合当前仍然有效的安排？只回答 A 或 B。\n'
                f'A：{choices[0]}\nB：{choices[1]}')
    prompt = prompt_for(tokenizer, transcript, question, deepseek)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > POSITIONS:
        raise ValueError('Choice prompt exceeded the validated position limit')
    ids = torch.tensor([tokens], device='cuda')
    output = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    chosen = output.logits[0, -1, labels].float()
    if not torch.isfinite(chosen).all():
        raise ValueError('Nonfinite choice logits')
    return (chosen[0] - chosen[1]).item()


def evaluate(model, tokenizer, deepseek):
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in ('A', 'B')]
    if any(len(tokens) != 1 for tokens in label_tokens) or label_tokens[0] == label_tokens[1]:
        raise ValueError('The labels must be distinct single tokens')
    labels = [tokens[0] for tokens in label_tokens]
    counts = {mode: Counter() for mode in ('all_turns', 'last_turn', 'no_turns')}
    rows = []
    for name, earlier, final, choices in PAIRS:
        contrasts = []
        for expected, last in enumerate(final):
            earlier_turns = list(enumerate(earlier, 1))
            final_turn = [(len(earlier) + 1, last)]
            transcripts = {'all_turns': earlier_turns + final_turn, 'last_turn': final_turn, 'no_turns': []}
            outcomes = {}
            for mode, transcript in transcripts.items():
                original = logit_margin(model, tokenizer, transcript, choices, labels, deepseek)
                switched = logit_margin(model, tokenizer, transcript, choices[::-1], labels, deepseek)
                averaged = (original - switched) / 2
                sign = 1 if expected == 0 else -1
                success = averaged * sign > 0
                counts[mode].update(ordered=sign * original > 0, switched=sign * switched < 0,
                                    averaged=success, order_consistent=original * switched < 0)
                outcomes[mode] = {'original': original, 'switched': switched, 'averaged': averaged,
                                  'correct': success}
            contrasts.append({'last': last, 'gold': expected, 'modes': outcomes})
        for mode in counts:
            counts[mode]['complete_pairs'] += all(row['modes'][mode]['correct'] for row in contrasts)
        rows.append({'name': name, 'earlier': earlier, 'choices': choices, 'contrasts': contrasts})
        print(name, [row['modes']['all_turns']['correct'] for row in contrasts], flush=True)
    return {'labels': labels, 'cases': len(PAIRS) * 2, 'summary': counts, 'rows': rows}


@torch.inference_mode()
def content_scores(model, tokenizer, transcript, choices, deepseek):
    prompt = prompt_for(tokenizer, transcript, '只写当前仍然有效的安排。', deepseek)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    values = []
    for choice in choices:
        full = tokenizer.encode(prompt + choice, add_special_tokens=False)
        if len(full) > POSITIONS or full[:len(prefix)] != prefix:
            raise ValueError('Answer text exceeds or changes the fixed prompt')
        count = len(full) - len(prefix)
        ids = torch.tensor([full], device='cuda')
        output = model(input_ids=ids, use_cache=False, logits_to_keep=count + 1)
        logits = output.logits[0, :-1].float().log_softmax(-1)
        average = logits.gather(-1, ids[0, -count:, None]).mean()
        if not torch.isfinite(average):
            raise ValueError('Nonfinite candidate likelihood')
        values.append(average.item())
    return values


def evaluate_content(model, tokenizer, deepseek):
    counts = {mode: Counter() for mode in ('all_turns', 'last_turn', 'no_turns', 'context_gain')}
    rows = []
    for name, earlier, final, choices in PAIRS:
        prior = content_scores(model, tokenizer, [], choices, deepseek)
        contrasts = []
        for expected, last in enumerate(final):
            earlier_turns = list(enumerate(earlier, 1))
            final_turn = [(len(earlier) + 1, last)]
            full = content_scores(model, tokenizer, earlier_turns + final_turn, choices, deepseek)
            values = {'all_turns': full, 'last_turn': content_scores(model, tokenizer, final_turn, choices, deepseek),
                      'no_turns': prior, 'context_gain': [value - bias for value, bias in zip(full, prior, strict=True)]}
            outcomes = {}
            for mode, scores in values.items():
                correct = scores[expected] > scores[1 - expected]
                counts[mode]['correct'] += correct
                outcomes[mode] = {'scores': scores, 'correct': correct}
            contrasts.append({'last': last, 'gold': expected, 'modes': outcomes})
        for mode in counts:
            counts[mode]['complete_pairs'] += all(row['modes'][mode]['correct'] for row in contrasts)
        rows.append({'name': name, 'earlier': earlier, 'choices': choices, 'contrasts': contrasts})
        print(name, [row['modes']['context_gain']['correct'] for row in contrasts], flush=True)
    return {'cases': len(PAIRS) * 2, 'summary': counts, 'rows': rows}


def main(args):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    destination = target / (f'{args.model}.json' if args.score == 'labels' else f'{args.model}-content.json')
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    assess = evaluate if args.score == 'labels' else evaluate_content
    if args.model == 'deepseek':
        chat = load_spora_for_eval(target)
        result = assess(chat.decoder, chat.tokenizer, True)
        result['runtime'] = chat.awake_individual
    else:
        if sha(args.checkpoint) != CHECKPOINT_SHA:
            raise ValueError('Unexpected SPORA checkpoint')
        base = load_qwen_streamed(args.donor, QWEN_SHA)
        tokenizer = AutoTokenizer.from_pretrained(args.donor, local_files_only=True, trust_remote_code=False)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        metadata, tensors = payload['metadata'], payload['tensors']
        if _digest(metadata, tensors) != payload['integrity_sha256']:
            raise ValueError('SPORA checkpoint integrity mismatch')
        model, organism, state = attach_control(base, metadata, tensors)
        individual = frozen_digest(organism, state)
        result = assess(model, tokenizer, False)
        if frozen_digest(organism, state) != individual:
            raise ValueError('Evaluation changed the SPORA state')
        result['runtime'] = {'individual_sha256': individual, 'checkpoint_sha256': CHECKPOINT_SHA,
                             'base_sha256': QWEN_SHA}
    store_json(destination, {**result, 'seconds': perf_counter() - started,
                             'peak_bytes': torch.cuda.max_memory_allocated()})
    print(result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', choices=('deepseek', 'qwen'), required=True)
    parser.add_argument('--score', choices=('labels', 'content'), default='labels')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--donor', type=Path)
    args = parser.parse_args()
    if args.model == 'qwen' and (args.checkpoint is None or args.donor is None):
        parser.error('--checkpoint and --donor are required for the Qwen control')
    main(args)
