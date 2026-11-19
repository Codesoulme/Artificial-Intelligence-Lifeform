"""Calibrate native replacement layers on short student-generated prefixes."""

import argparse
import math
import random
from pathlib import Path

from audit_awake_language_layers import QUESTIONS, REFERENCES, load_layer_pair
from deepseek_output_aligned_memory import MODEL, SOURCE, sha, torch, write
from spora.language_training import atomic_save


TRAIN = (
    ('暖茶敞开放在室内，为什么会变凉？', '茶水会向更冷的空气散热。'),
    ('推荐一种不加糖的饮品，只推荐一种。', '可以喝无糖气泡水。'),
    ('四盒粉笔，每盒三根，总共多少根？', '一共十二根。'),
    ('把「她最爱读什么书」翻译成英文。', 'What books does she like to read most?'),
    ('天凉了，出门可以带什么衣服？', '可以带一件保暖外套。'),
    ('给我恰好两个睡前放松的建议。', '做几分钟深呼吸，或者读几页轻松的书。'),
    ('你知道我今天中午吃了什么吗？', '我不知道，你还没有告诉我。'),
    ('只回答「收到」二字。', '收到'),
    ('普通玻璃为什么能透光？用一句话回答。', '普通玻璃对可见光的吸收很少，因此大部分可见光可以通过。'),
    ('用一句话感谢帮忙搬箱子的朋友。', '谢谢你帮我搬箱子。'),
    ('从九个苹果里拿走四个，还剩几个？', '还剩五个。'),
    ('Translate into Chinese: Where did she go?', '她去了哪里？'),
)
SEED = 57013


def encode_cases(tokenizer, pairs):
    rows = []
    for question, answer in pairs:
        text = tokenizer.apply_chat_template([dict(role='user', content=question)], tokenize=False,
                                              add_generation_prompt=True)
        prompt = tokenizer.encode(text + '</think>\n\n', add_special_tokens=False)
        target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        rows.append((question, prompt, target))
    return rows


def switch(decoder, layers):
    for index, block in layers.items():
        decoder.base.model.layers[index] = block


@torch.no_grad()
def generate(decoder, tokenizer, prompt, limit):
    ids = torch.tensor([prompt], device='cuda')
    output = decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=limit,
                              use_cache=True, do_sample=False, top_p=None, temperature=None,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return output[0, len(prompt):].tolist()


def answer_logits(decoder, prompt, tails):
    ids = torch.tensor([prompt + tails[:-1]], device='cuda')
    return decoder(input_ids=ids, use_cache=False, logits_to_keep=len(tails)).logits[0].float()


@torch.no_grad()
def teacher_targets(decoder, cases):
    return [answer_logits(decoder, prompt, tails).log_softmax(-1).cpu() for _, prompt, tails in cases]


def divergence(logits, teacher):
    return torch.nn.functional.kl_div(logits.log_softmax(-1), teacher.to(logits.device),
                                      log_target=True, reduction='batchmean')


@torch.no_grad()
def evaluate(decoder, tokenizer, cases, teacher):
    rows = []
    for (question, prompt, target), base in zip(cases, teacher, strict=True):
        logits = answer_logits(decoder, prompt, target)
        nll = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device='cuda')).item()
        kl = divergence(logits, base).item()
        tokens = generate(decoder, tokenizer, prompt, 96)
        rows.append(dict(question=question, nll=nll, kl=kl, eos=tokens[-1] == tokenizer.eos_token_id,
                         answer=tokenizer.decode(tokens, skip_special_tokens=True).strip()))
    return rows


def main(output, steps, learning_rate, reference_weight):
    protected = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors')}
    torch.manual_seed(SEED)
    decoder, tokenizer, originals, replacements, _, _ = load_layer_pair()
    train = encode_cases(tokenizer, TRAIN)
    held = encode_cases(tokenizer, tuple(zip(QUESTIONS, REFERENCES, strict=True)))
    if {question for question, _, _ in train} & {question for question, _, _ in held}:
        raise ValueError('Overlapping prompts')
    replay = [(question, prompt, generate(decoder, tokenizer, prompt, 16)) for question, prompt, _ in train]
    switch(decoder, originals)
    replay_targets = teacher_targets(decoder, replay)
    held_targets = teacher_targets(decoder, held)
    switch(decoder, replacements)
    print('Captured teacher and student prefixes', flush=True)
    before = evaluate(decoder, tokenizer, held, held_targets)
    editable = {}
    for index, block in replacements.items():
        editable[index] = {
            name: parameter.requires_grad_(True)
            for name, parameter in block.named_parameters()
            if name.startswith(('block.norm1.', 'block.norm2.')) or name == 'block.ff.2.bias'
        }
        if not editable[index]:
            raise ValueError('Replacement has no calibration parameters')
    parameters = [parameter for group in editable.values() for parameter in group.values()]
    initial = {index: {name: parameter.detach().clone() for name, parameter in group.items()}
               for index, group in editable.items()}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    generator = random.Random(SEED)
    losses = []
    for _ in range(steps):
        index = generator.randrange(len(train))
        _, prompt, target = train[index]
        _, _, prefix = replay[index]
        optimizer.zero_grad(set_to_none=True)
        replay_loss = divergence(answer_logits(decoder, prompt, prefix), replay_targets[index])
        replay_loss.backward()
        supervised = torch.nn.functional.cross_entropy(answer_logits(decoder, prompt, target),
                                                       torch.tensor(target, device='cuda'))
        (reference_weight * supervised).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(replay_kl=replay_loss.item(), reference_nll=supervised.item()))
        if len(losses) % 8 == 0:
            print('Calibration steps', len(losses), losses[-1], flush=True)
    after = evaluate(decoder, tokenizer, held, held_targets)
    if any(sha(path) != previous for path, previous in protected.items()):
        raise ValueError('Source model or checkpoint changed during training')
    delta = {index: {name: parameter.detach().cpu() - initial[index][name].cpu()
                     for name, parameter in group.items()} for index, group in editable.items()}
    atomic_save(dict(source_sha256=protected[SOURCE], delta=delta), output / 'delta.pt')
    summary = {name: dict(nll=sum(r['nll'] for r in rows) / len(rows), kl=sum(r['kl'] for r in rows) / len(rows),
                          eos=sum(r['eos'] for r in rows)) for name, rows in (('before', before), ('after', after))}
    write(output / 'result.json', dict(summary=summary, training=TRAIN, steps=steps, learning_rate=learning_rate,
                                       reference_weight=reference_weight, losses=losses, before=before,
                                       after=after, sources={str(p): h for p, h in protected.items()},
                                       delta_sha256=sha(output / 'delta.pt')))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--reference-weight', type=float, default=.2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if not all(math.isfinite(value) and value > 0 for value in (args.learning_rate, args.reference_weight)):
        parser.error('Learning rate and reference weight must be finite and positive')
    args.output.mkdir()
    main(args.output, args.steps, args.learning_rate, args.reference_weight)
