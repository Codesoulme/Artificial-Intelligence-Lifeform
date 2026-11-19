"""Check a replacement calibration delta against fresh prompts and persisted memory."""

import argparse
from pathlib import Path

from deepseek_awake_recall_intent import DEFAULT_DATABASE
from deepseek_continuous_memory_chat import EARLY_TIMING_THRESHOLD, load_chat
from deepseek_output_aligned_memory import SOURCE, read, sha, torch, write
from spora.awake_recall_intent import OrdinaryIntent, SelfNameIntent
from train_awake_replacement_replay import encode_cases, generate


QUESTIONS = (
    ('把「他最想去哪里」翻译成英文。', 'Where does he most want to go?'),
    ('五张桌子，每桌摆三把椅子，一共有几把椅子？', '一共有十五把椅子。'),
    ('为什么冬天金属门把手摸起来比木头凉？', '金属导热较快，会更快带走手上的热量。'),
    ('给我一个放松肩膀的方法，只说一个。', '轻轻转动肩膀，慢慢放松。'),
    ('我还没有说过我的早餐，你知道我吃了什么吗？', '我不知道。'),
    ('只回复两个字：明白。', '明白'),
)
MEMORY = (
    ('齐宁平常最爱喝什么？', '青杏冰茶'),
    ('这次要填履历，齐宁当时做的是哪份工作？', '织机检修师'),
    ('齐宁那次去了哪里？', '新荷档案馆'),
    ('温谨平常最爱喝什么？', '桂花乌龙'),
    ('沈亦平常最爱喝什么？', '香橙冰茶'),
    ('上一次记录里，沈亦平常最爱喝什么？', '甘蔗乌龙茶'),
    ('你叫什么名字？', '孢子'),
)
ALLOWED = {'block.norm1.weight', 'block.norm1.bias', 'block.norm2.weight', 'block.norm2.bias', 'block.ff.2.bias'}


@torch.no_grad()
def apply_delta(decoder, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['source_sha256'] != sha(SOURCE):
        raise ValueError('Delta is bound to a different source checkpoint')
    metadata = torch.load(SOURCE, map_location='cpu', weights_only=True)['metadata']
    indices = {row['index'] for row in metadata['replacements']}
    if set(payload['delta']) != indices:
        raise ValueError('Delta replacement indices differ from the checkpoint')
    changes = []
    for index, tensors in payload['delta'].items():
        parameters = dict(decoder.base.model.layers[index].named_parameters())
        if set(tensors) != ALLOWED:
            raise ValueError('Invalid delta parameter names')
        for name, delta in tensors.items():
            parameter = parameters[name]
            if delta.shape != parameter.shape or delta.dtype != parameter.dtype or not bool(torch.isfinite(delta).all()):
                raise ValueError(f'Invalid delta tensor for layer {index} {name}')
            changes.append((parameter, delta))
    for parameter, delta in changes:
        parameter.add_(delta.to(parameter.device))


def evaluate(chat, cases):
    ordinary = []
    for question, prompt, _ in cases:
        tokens = generate(chat.decoder, chat.tokenizer, prompt, 96)
        ordinary.append(dict(question=question, answer=chat.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                             eos=tokens[-1] == chat.tokenizer.eos_token_id))
    memory = []
    for question, expected in MEMORY:
        row = chat.answer(question, timing_threshold=EARLY_TIMING_THRESHOLD)
        correct = row['action'] == 'read' and row['response'].rstrip('」。').endswith(expected)
        memory.append(dict(question=question, expected=expected, correct=correct, **row))
    return dict(ordinary=ordinary, memory=memory)


def main(delta_path, database, output, profile, ordinary_head):
    digest = sha(database)
    delta_digest = sha(delta_path)
    if read(delta_path.parent / 'result.json')['delta_sha256'] != delta_digest:
        raise ValueError('Delta digest differs from the training report')
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    width = chat.decoder.base.lm_head.weight.shape[1]
    chat.self_name = SelfNameIntent.load_name(profile / 'self-name.pt', width, chat.recall_fingerprint,
                                             chat.bank_specs['drink', 1])
    chat.ordinary_intent = OrdinaryIntent.load(ordinary_head, width, chat.recall_fingerprint)
    cases = encode_cases(chat.tokenizer, QUESTIONS)
    before = evaluate(chat, cases)
    apply_delta(chat.decoder, delta_path)
    after = evaluate(chat, cases)
    if sha(database) != digest or sha(delta_path) != delta_digest:
        raise ValueError('Protected database or delta changed during verification')
    summary = dict(ordinary_eos_before=sum(row['eos'] for row in before['ordinary']),
                   ordinary_eos_after=sum(row['eos'] for row in after['ordinary']),
                   memory_values_before=sum(row['correct'] for row in before['memory']),
                   memory_values_after=sum(row['correct'] for row in after['memory']),
                   memory_unchanged=sum((a['action'], a['response']) == (b['action'], b['response'])
                                        for a, b in zip(before['memory'], after['memory'], strict=True)))
    write(output, dict(summary=summary, before=before, after=after, delta_sha256=delta_digest, database_sha256=digest))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('delta', type=Path)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--ordinary-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    main(args.delta, args.database, args.output, args.profile, args.ordinary_head)
