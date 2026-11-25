"""Test fixed public Chinese tasks and replay of correct revision tokens."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_local_language_boundary import BUDGET, assess
from scripts.audit_spora_ordinary_reply_residual import ReplyResidual, collect, enabled, fit, lessons
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


PUBLIC_SHA = '65400edb71326ad8e69474701ab14252c4d3d2e9b8b49643716d897967d6636e'
PUBLIC_CASES = ((7, 1), (7, 2), (12, 1), (12, 3), (15, 1), (15, 3), (18, 1), (18, 3),
                (22, 1), (22, 3), (28, 1), (28, 3))
PUBLIC_BUDGET = 128
ANCHORS = (
    ('明天六点在南廊等你，带好借阅证。', '把明天改为星期四。其他信息保留。', '星期四六点在南廊等你，带好借阅证。'),
    ('下午拿两把蓝伞，送到旧门房。', '蓝伞改成黑伞，数量和地点保持。', '下午拿两把黑伞，送到旧门房。'),
    ('午休之后在小广场见，穿灰外套。', '地点改为长桥下，其他不变。', '午休之后在长桥下见，穿灰外套。'),
    ('星期三寄维修单，抄送李老师。', '寄出日期改为星期五，其余照旧。', '星期五寄维修单，抄送李老师。'),
    ('后天晚上八点开会，带上名单。', '把八点改为九点，保留其他信息。', '后天晚上九点开会，带上名单。'),
    ('买两罐绿茶和一盒糕点。', '绿茶改为三罐，糕点照旧。', '买三罐绿茶和一盒糕点。'),
    ('周一在旧教室取绘画册和小刀。', '不用带小刀，其他不动。', '周一在旧教室取绘画册。'),
    ('把蓝色长椅放进仓库，贴上白标签。', '把白标签换成红标签，其余条件不变。', '把蓝色长椅放进仓库，贴上红标签。'),
    ('请给凌阿姨发照片，晚上十点之前。', '只改截止时间为晚上七点。', '请在晚上七点之前给凌阿姨发照片。'),
    ('周六在东桥口集合，携带相机。', '集合改到茶园门口，相机以及时间保持不变。', '周六在茶园门口集合，携带相机。'),
    ('帮小陈拿三张本子，送去演播室。', '把数量改成五张，人物和地点不变。', '帮小陈拿五张本子，送去演播室。'),
    ('这次聚会选在周四晚饭后，记得带拼图。', '聚会改成周日，其他保持。', '这次聚会选在周日晚饭后，记得带拼图。'),
)
REVISIONS = (
    ('日期1', '小唐，星期一下午在西厅见，带好合同。', '日期改成星期二，仍是下午，地点和合同都保持。',
     ('星期二|周二', '下午', '西厅', '合同'), ('星期一|周一',)),
    ('日期2', '请在周五下午给冯老师发路演资料。', '把发送时间改到周一早上，其他保持，一句话。',
     ('周一|星期一', '早上|上午', '冯老师', '路演资料'), ('周五|星期五|下午',)),
    ('数量', '请在石桥门口给小周送六份试卷。', '改成四份，地点和收件人保持。直接给我通知。',
     ('四份|4份', '石桥门口', '小周', '试卷'), ('六份|6份',)),
    ('地点', '明早十点在北角亭见，带好竹篮。', '改去书市门口，时间和东西都保留。',
     ('明早|明天早', '十点|10点', '书市门口', '竹篮'), ('北角亭',)),
    ('物件', '提醒乐乐周二在社区大厅带雨伞。', '雨伞改成长筒雨鞋，人物、日子和地点保留。',
     ('乐乐', '周二|星期二', '社区大厅', '长筒雨鞋'), ('雨伞',)),
    ('取消', '出门前提醒我拔电源和收毛巾。', '只留拔电源，把另外一项删掉。',
     ('拔.{0,3}(?:电源|插头)',), ('毛巾',)),
    ('保留两项', '午饭后陪小梁去河岸照相，给他带手套。', '取消照相，保留时间、去哪里和带手套。',
     ('午饭后', '小梁', '河岸', '手套'), ('照相|拍照',)),
    ('时间', '下周四六点半在体育场大门借球拍。', '六点半改成八点整，保留日期、地点、借球拍。',
     ('周四|星期四', '八点|8点', '体育场大门', '借.{0,4}球拍'), ('六点半|6点半',)),
)


def history_for(previous):
    return ({'role': 'user', 'content': '帮我写一条简短消息。'}, {'role': 'assistant', 'content': previous})


def public_rows(path):
    if base.sha(path) != PUBLIC_SHA:
        raise ValueError('Expected the pinned Chinese FollowBench format file')
    data = json.loads(path.read_text(encoding='utf-8'))
    if len(data) != 180:
        raise ValueError('Unexpected public case count')
    found = {(row['example_id'], row['level']): row['instruction'] for row in data}
    return [(group, level, found[group, level]) for group, level in PUBLIC_CASES]


def ordered(text, *patterns):
    at = 0
    for pattern in patterns:
        hit = re.search(pattern, text[at:], re.IGNORECASE)
        if not hit:
            return False
        at += hit.end()
    return True


def table(text, columns):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in text.splitlines() if '|' in line]
    return rows if len(rows) > 2 and all(len(row) == columns and all(row) for row in rows) else []


def paragraphs(text):
    return [part.strip() for part in re.split(r'\n\s*\n', text.strip()) if part.strip()]


def sentences(text):
    return [part.strip() for part in re.split(r'[。！？!?]', text) if part.strip()]


def public_checks(group, level, answer):
    plain = answer.strip()
    length_ok = True
    if group == 7:
        content = bool(re.search(r'胶片|胶卷', plain) and re.search(r'颗粒|色彩|怀旧|质感|过程|体验', plain))
        structure = ordered(plain, r'引言|前言', r'主要论点|核心观点|论点', r'结论|总结')
        if level == 2:
            sections = re.split(r'引言|前言|主要论点|核心观点|论点|结论|总结', plain)
            length_ok = all(len(segment) <= 200 for segment in sections)
    elif group == 12:
        content = bool(re.search(r'PyTorch|TensorFlow|Keras|Paddle|JAX', plain, re.IGNORECASE))
        structure = ordered(plain, r'PyTorch|TensorFlow|Keras|Paddle|JAX', r'功能|特点|优势|作用', r'应用|用例|用途|使用场景')
        if level == 3:
            structure = structure and bool(re.search(r'^\s*(?:[-*]|\d+[.、])\s*\S', plain, re.MULTILINE))
    elif group == 15:
        content = bool(re.search(r'问题|目标', plain) and re.search(r'拆|分解|查|验证|尝试|记录', plain))
        if level == 3:
            rows = table(plain, 3)
            structure = bool(rows and rows[0] == ['策略', '步骤', '详细信息'])
        else:
            structure = ordered(plain, r'总体|思路|方法|策略|原则', r'步骤|技巧|具体')
    elif group == 18:
        content = bool(re.search(r'三文鱼|鲑鱼|鱼柳|鱼片', plain) and re.search(r'煎|锅|预热', plain))
        if level == 3:
            rows = table(plain, 2)
            structure = bool(rows and '主要' in rows[0][0] and re.search(r'次要|对应', rows[0][1]))
            length_ok = bool(rows) and all(len(re.sub(r'\s', '', row[0])) <= 10 for row in rows[2:])
        else:
            structure = bool(re.search(r'主要|步骤', plain) and re.search(r'^ {2,}[-*]|子步骤|次要|具体', plain, re.MULTILINE))
    elif group == 22:
        content = bool(re.search(r'木桌|木质|木材', plain) and re.search(r'擦|保养|避免|防', plain))
        parts = paragraphs(plain)
        structure = len(parts) == 3
        if level == 3:
            length_ok = all(1 <= len(sentences(part)) <= 3 and all(len(text) > 20 for text in sentences(part)) for part in parts)
    elif group == 28:
        content = bool(re.search(r'冲绳|那霸|首里|美丽海|万座毛|国际通', plain))
        titles = re.findall(r'<h[1-6](?:\s+[^>]*)?>(.*?)</h[1-6]>', plain, re.IGNORECASE | re.DOTALL)
        structure = bool(re.search(r'<(?:html|body|h[1-6]|p)(?:\s|>)', plain, re.IGNORECASE))
        if level == 3:
            structure = bool(titles and re.search(r'<p(?:\s+[^>]*)?>.+?</p>', plain, re.IGNORECASE | re.DOTALL))
            length_ok = bool(titles) and all(len(re.sub(r'<[^>]+>|\s', '', title)) <= 6 for title in titles)
    else:
        raise ValueError(f'No public scoring rule for {group}')
    return {'content': content, 'structure': structure, 'length': length_ok}


@torch.no_grad()
def anchor_distributions(model, tokenizer):
    kept, teacher = [], []
    for previous, question, answer in ANCHORS:
        ids = closed_think_inputs(tokenizer, question, model.device, history=history_for(previous))['input_ids']
        truth = torch.tensor(tokenizer.encode(answer, add_special_tokens=False), device=model.device)
        combined = torch.cat((ids, truth[:-1].unsqueeze(0)), dim=1)
        hidden = model.model(input_ids=combined, use_cache=False).last_hidden_state[0, -len(truth):]
        scores = model.lm_head(hidden).float()
        meaningful = torch.tensor([any(char.isalnum() for char in tokenizer.decode([token]))
                                   for token in truth.tolist()], device=model.device)
        matches = (scores.argmax(-1) == truth) & meaningful
        kept.append(hidden[matches])
        teacher.append(scores[matches].softmax(-1))
    combined = torch.cat(kept)
    if not len(combined):
        raise ValueError('The frozen model supplied no matching preservation positions')
    return combined, torch.cat(teacher)


def state_adapter(model, weights):
    adapter = ReplyResidual(model.config.hidden_size)
    adapter.load_state_dict(weights)
    return adapter.eval().to(model.device)


def run_case(model, tokenizer, modes, question, history, budget):
    outputs = {}
    for name, adapter in modes.items():
        if adapter is None:
            output = generate(model, tokenizer, question, history=history, max_new_tokens=budget)
        else:
            with enabled(model, adapter):
                output = generate(model, tokenizer, question, history=history, max_new_tokens=budget)
        tokens = output['token_ids']
        outputs[name] = {**output, 'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id)}
    return outputs


def summarize(rows):
    totals = {split: Counter() for split in ('public', 'revision')}
    for row in rows:
        counts = totals[row['split']]
        baseline = row['outputs']['base']['passed']
        for mode, answer in row['outputs'].items():
            counts.update({f'{mode}_pass': answer['passed'], f'{mode}_eos': answer['eos'],
                           f'{mode}_retained': baseline and answer['passed'],
                           f'{mode}_gained': not baseline and answer['passed'],
                           f'{mode}_lost': baseline and not answer['passed']})
    return totals


@torch.no_grad()
def anchor_checks(model, modes, hidden, teacher):
    metrics = {}
    first_tokens = teacher.argmax(-1)
    for mode, adapter in modes.items():
        values = hidden if adapter is None else adapter(hidden)
        scores = model.lm_head(values).float().log_softmax(-1)
        metrics[mode] = {'kl': float(F.kl_div(scores, teacher, reduction='batchmean')),
                         'top1_matches': int((scores.argmax(-1) == first_tokens).sum())}
    return metrics


def verify_outputs(model, tokenizer, modes, prior, public_count):
    checked = []
    for index in (0, public_count, public_count + 2):
        row = prior['rows'][index]
        history = history_for(row['previous']) if row['split'] == 'revision' else ()
        budget = BUDGET if history else PUBLIC_BUDGET
        actual = run_case(model, tokenizer, modes, row['question'], history, budget)
        if any(output['token_ids'] != row['outputs'][mode]['token_ids'] for mode, output in actual.items()):
            raise ValueError('A reloaded residual changed a saved output')
        checked.append({'split': row['split'], 'group': row['group'], 'modes': list(actual)})
    return checked


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    reference = json.loads((args.reference / 'reload.json').read_text(encoding='utf-8'))
    old_path = args.reference / 'residuals.pt'
    adapter_path = args.output / 'preserving.pt'
    model_path = base.MODEL / 'model.safetensors'
    sources = {str(path): base.sha(path) for path in (args.data, old_path, model_path, base.MODEL / 'tokenizer.json',
                                                     args.reference / 'reload.json', Path(__file__))}
    if (sources[str(old_path)] != reference['head_sha256'] or sources[str(model_path)] != base.WEIGHT_SHA
            or reference['sources'][str(model_path)] != sources[str(model_path)]):
        raise ValueError('The older adaptation and its frozen base do not match')
    if prior and (any(prior['sources'].get(path) != digest for path, digest in sources.items()
                      if path != str(Path(__file__))) or base.sha(adapter_path) != prior['adapter_sha256']):
        raise ValueError('A saved comparison or adapter changed before reload')
    public = public_rows(args.data)
    training = lessons()
    held_questions = {question for _, _, question in public} | {question for _, _, question, _, _ in REVISIONS}
    if held_questions & ({question for question, _, _ in training} | {question for _, question, _ in ANCHORS}):
        raise ValueError('A held-out question appears in the training material')
    model = load_full_streamed().requires_grad_(False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    anchor_hidden, teacher = anchor_distributions(model, tokenizer)
    print('preservation positions', len(anchor_hidden), flush=True)
    old = state_adapter(model, torch.load(old_path, map_location='cpu', weights_only=True)['supervised'])
    if prior:
        if len(anchor_hidden) != prior['anchor_positions']:
            raise ValueError('The preservation positions changed after reload')
        preserving = state_adapter(model, torch.load(adapter_path, map_location='cpu', weights_only=True))
        modes = {'base': None, 'supervised': old, 'preserving': preserving}
        metrics = anchor_checks(model, modes, anchor_hidden, teacher)
        checked = verify_outputs(model, tokenizer, modes, prior, len(public))
        if base.sha(adapter_path) != prior['adapter_sha256'] or any(base.sha(Path(path)) != digest for path, digest in sources.items()):
            raise ValueError('A protected input changed in the reload')
        store_json(args.output / 'anchor-check.json', {'anchor_positions': len(anchor_hidden), 'metrics': metrics,
                                                       'verified': checked, 'seconds': perf_counter() - started})
        print(metrics, flush=True)
        return
    hidden, labels, mistakes = collect(model, tokenizer, training)
    preserving = fit(model, hidden, labels, mistakes, steps=reference['steps'], contrastive=False,
                     preservation=(anchor_hidden, teacher))
    atomic_save(preserving.state_dict(), adapter_path)
    modes = {'base': None, 'supervised': old, 'preserving': preserving}
    rows = []
    for group, level, question in public:
        outputs = run_case(model, tokenizer, modes, question, (), PUBLIC_BUDGET)
        for output in outputs.values():
            checks = public_checks(group, level, output['answer'])
            output.update(checks=checks, passed=output['eos'] and all(checks.values()))
        rows.append({'split': 'public', 'group': group, 'level': level, 'question': question, 'outputs': outputs})
        print('public', group, level, {name: row['passed'] for name, row in outputs.items()}, flush=True)
    for name, previous, question, required, forbidden in REVISIONS:
        outputs = run_case(model, tokenizer, modes, question, history_for(previous), BUDGET)
        for output in outputs.values():
            checks = assess(output['answer'], output['eos'], required, (*forbidden, r'\n|以下'), 65)
            output.update(checks=checks, passed=checks['pass'])
        rows.append({'split': 'revision', 'group': name, 'previous': previous, 'question': question, 'outputs': outputs})
        print('revision', name, {mode: row['passed'] for mode, row in outputs.items()}, flush=True)
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A frozen input changed during evaluation')
    result = {'sources': sources, 'rows': rows, 'totals': summarize(rows), 'anchor_positions': len(anchor_hidden),
              'training_tokens': len(labels), 'steps': reference['steps'], 'public_budget': PUBLIC_BUDGET,
              'adapter_sha256': base.sha(adapter_path), 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    store_json(args.output / 'result.json', result)
    print(result['totals'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'reference', 'output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--verify', action='store_true')
    main(cli.parse_args())
