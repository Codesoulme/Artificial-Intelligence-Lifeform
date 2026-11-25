"""Compare two frozen-DeepSeek reply residuals on ordinary local chat."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_local_language_boundary import BUDGET, HELD, assess, cases_from_smoke, direct
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


SEED = 32643
TRANSLATIONS = (
    ('先等两分钟', 'Wait two minutes first.'), ('今天就到这里', "Let's stop here for today."),
    ('可以换一把椅子吗', 'Can we use another chair?'), ('我还在等答复', "I'm still waiting for a reply."),
    ('请发给收件人', 'Please send it to the recipient.'), ('别把钥匙落在车里', "Don't leave the keys in the car."),
    ('这份先放到明天', "Let's leave this one until tomorrow."), ('还有别的选择吗', 'Is there another option?'),
)
SHORT_ACTIONS = (
    ('忘了带笔', '向身边的人借一支笔。'), ('包里塞满皱纸', '把包里的废纸拿出来。'),
    ('鞋子挡住门口', '把门口的鞋挪到墙边。'), ('网页突然卡住', '先刷新当前页面。'),
    ('手机电量只剩一点', '先给手机接上充电线。'), ('刚打翻了清水', '先用抹布吸掉桌上的水。'),
    ('衣服堆在床上', '把待洗衣服放进洗衣篮。'), ('抽屉里的线缠住了', '先把所有线拿出抽屉。'),
)
REVISIONS = (
    ('下午见面，带资料和笔。', '取消带笔，只发一条提醒。', '下午见面，记得带资料。'),
    ('出门前带伞，顺便关灯。', '去掉伞，只留关灯。', '出门前记得关灯。'),
    ('下周二午饭在西门集合，带好工牌。', '改到周三，其他保持。', '周三午饭在西门集合，带好工牌。'),
    ('上午十点到南门，先给小陈打电话。', '保留十点和南门，不打电话。', '上午十点到南门。'),
    ('周末去图书馆，带借阅卡和水壶。', '只保留借阅卡这一项。', '去图书馆记得带借阅卡。'),
    ('请同事今天发清单和收据。', '只催收据，把时间改到明早。', '请在明早把收据发给我。'),
    ('周五中午在车站碰面，带运动鞋。', '把碰面时间改成周四中午，别改地点或物品。', '周四中午在车站见，记得带运动鞋。'),
    ('包裹改送公司前台，下午签收。', '只确认地点，别提具体时间。', '包裹改送公司前台。'),
)
FACTS = (
    ('灯关了为什么屋里暗下来？', '因为灯停止发光，室内少了一个光源。'),
    ('车票显示候补，能确认已经买到了吗？', '不能，候补还没有确认出票。'),
    ('冷藏的水一拿出来为什么瓶壁出现水珠？', '空气里的水蒸气碰到冷瓶壁凝结成水滴。'),
    ('供应商说材料未发货，能确定已经到站了吗？', '不能，材料还未发货。'),
    ('毛巾为什么通常在风里干得快？', '风带走湿空气，毛巾里的水分蒸发更快。'),
    ('门没锁，就能确定有人进去过吗？', '不能，门没锁只说明可以打开。'),
    ('有树荫的地方为什么常更凉快？', '树冠遮住阳光，地面吸收的热量更少。'),
    ('饭店显示暂停营业，能知道它哪天重新开门吗？', '不能，还需要查看恢复营业的通知。'),
)


class ReplyResidual(nn.Module):
    def __init__(self, width, rank=16):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        delta = self.up(F.gelu(self.down(value.float())))
        return value + delta.to(value.dtype)


def lessons():
    rows = [(f'把「{text}」译成英语，只给译文。', (), answer) for text, answer in TRANSLATIONS]
    rows += [(f'现在{text}。告诉我一个可立即做的动作，最多一句。', (), answer) for text, answer in SHORT_ACTIONS]
    rows += [(question, ({'role': 'user', 'content': '帮我写一条简短通知。'},
                         {'role': 'assistant', 'content': old}), answer) for old, question, answer in REVISIONS]
    rows += [(question + '只用一句话。', (), answer) for question, answer in FACTS]
    return rows


@torch.no_grad()
def collect(model, tokenizer, rows):
    features, labels, divergence = [], [], []
    offset = 0
    for question, history, answer in rows:
        prompt = closed_think_inputs(tokenizer, question, model.device, history=history)['input_ids']
        target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        truth = torch.tensor(target, device=model.device)
        joined = torch.cat((prompt, truth[:-1].unsqueeze(0)), dim=1)
        hidden = model.model(input_ids=joined, use_cache=False).last_hidden_state[0, -len(target):]
        predicted = model.lm_head(hidden).argmax(-1)
        misses = (predicted != truth).nonzero().flatten()
        if misses.numel():
            index = int(misses[0])
            divergence.append((offset + index, int(predicted[index])))
        features.append(hidden.detach())
        labels.append(truth)
        offset += len(target)
    return torch.cat(features), torch.cat(labels), torch.tensor(divergence, device=model.device)


def fit(model, features, labels, mistakes, *, steps, contrastive):
    torch.manual_seed(SEED)
    adapter = ReplyResidual(model.config.hidden_size).to(model.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(SEED)
    for index in range(steps):
        positions = torch.randint(len(labels), (6,), generator=generator).to(model.device)
        pair = mistakes[torch.randint(len(mistakes), (2,), generator=generator).to(model.device)]
        positions = torch.cat((positions, pair[:, 0]))
        y = labels[positions]
        scores = model.lm_head(adapter(features[positions])).float()
        loss = F.cross_entropy(scores, y)
        if contrastive:
            selected = scores[-2:].gather(1, y[-2:, None]).squeeze(1)
            rejected = scores[-2:].gather(1, pair[:, 1:]).squeeze(1)
            loss = loss + .3 * F.softplus(1 - selected + rejected).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if index in (0, steps - 1):
            print('fit', contrastive, index + 1, float(loss.detach()), flush=True)
    return adapter.eval()


@contextmanager
def enabled(model, adapter):
    handle = model.model.norm.register_forward_hook(lambda module, args, value: adapter(value))
    try:
        yield
    finally:
        handle.remove()


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    first = json.loads((args.output / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    script = str(Path(__file__))
    sources = {str(path): base.sha(path) for path in (args.smoke, args.diagnostic, Path(script),
                                                     base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json')}
    if first and any(first['sources'].get(path) != digest for path, digest in sources.items() if path != script):
        raise ValueError('The saved residuals came from different input artifacts')
    prior = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    smoke = json.loads(args.smoke.read_text(encoding='utf-8'))
    cases = cases_from_smoke(smoke) + [(*row, None) for row in HELD]
    if any(prior['sources'].get(path) != digest for path, digest in sources.items() if path in prior['sources']):
        raise ValueError('The prior generation and current inputs do not match')
    model = load_full_streamed().requires_grad_(False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    training = lessons()
    if len(prior['rows']) != len(cases) or {row[0] for row in training} & {row[1] for row in cases}:
        raise ValueError('Expected matching prior cases separated from training')
    head_path = args.output / 'residuals.pt'
    if first:
        weights = torch.load(head_path, map_location='cpu', weights_only=True)
        adapters = {}
        for mode in ('supervised', 'contrastive'):
            adapter = ReplyResidual(model.config.hidden_size)
            adapter.load_state_dict(weights[mode])
            adapters[mode] = adapter.eval().to(model.device)
        tokens, divergences = first['supervised_tokens'], first['first_divergences']
    else:
        hidden, truth, mistakes = collect(model, tokenizer, training)
        adapters = {mode: fit(model, hidden, truth, mistakes, steps=args.steps, contrastive=mode == 'contrastive')
                    for mode in ('supervised', 'contrastive')}
        atomic_save({mode: adapter.state_dict() for mode, adapter in adapters.items()}, head_path)
        tokens, divergences = len(truth), len(mistakes)
    head_digest = base.sha(head_path)
    totals, rows = Counter(), []
    for case, earlier in zip(cases, prior['rows'], strict=True):
        name, question, history, required, forbidden, max_chars, _ = case
        messages = [{'role': role, 'content': text} for role, text in history]
        if (name, question, messages) != (earlier['name'], earlier['question'], earlier['history']):
            raise ValueError('A held-out prompt or history changed')
        baseline = dict(earlier['modes']['direct'])
        baseline['checks'] = assess(baseline['answer'], baseline['eos'], required, forbidden, max_chars)
        results = {'base': baseline}
        for mode, adapter in adapters.items():
            with enabled(model, adapter):
                output = direct(model, tokenizer, question, messages)
            output['checks'] = assess(output['answer'], output['eos'], required, forbidden, max_chars)
            results[mode] = output
        for mode, output in results.items():
            totals.update({f'{mode}_pass': output['checks']['pass'], f'{mode}_eos': output['eos']})
        print(name, {mode: row['checks']['pass'] for mode, row in results.items()}, flush=True)
        rows.append({'name': name, 'question': question, 'modes': results})
    repeat = direct(model, tokenizer, cases[0][1], prior['rows'][0]['history'])
    if repeat['token_ids'] != prior['rows'][0]['modes']['direct']['token_ids']:
        raise ValueError('Disabling the residual changed the frozen base path')
    if first:
        for result, expected in zip(rows, first['rows'], strict=True):
            if result['name'] != expected['name'] or result['question'] != expected['question']:
                raise ValueError('A reloaded comparison task changed')
            if any(result['modes'][mode]['token_ids'] != expected['modes'][mode]['token_ids'] for mode in adapters):
                raise ValueError('A saved residual did not reproduce its generation')
    if base.sha(head_path) != head_digest or any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A protected source changed')
    store_json(args.output / ('reload.json' if first else 'result.json'),
               {'sources': sources, 'rows': rows, 'totals': totals, 'head_sha256': head_digest,
                'training_examples': len(training), 'supervised_tokens': tokens, 'first_divergences': divergences,
                'steps': first['steps'] if first else args.steps,
                'parameters_per_arm': sum(p.numel() for p in adapters['supervised'].parameters()),
                'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})
    print(dict(totals), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('smoke', 'diagnostic', 'output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--steps', type=int, default=192)
    cli.add_argument('--verify', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('At least one gradient step is required')
    main(args)
