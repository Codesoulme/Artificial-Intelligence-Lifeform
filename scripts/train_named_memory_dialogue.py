"""Train ordinary-only residuals on multi-turn replies and contrast stale histories."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.audit_local_named_chat import CASES, FRESH_CASES, ModelRuntime, run
from scripts.audit_qwen_packed_reply_20261122 import HELD, evaluate, records
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.spora_named_memory_chat import LocalChat, MODEL, load_full_streamed, sha
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.pretrained_attention import SPORAPretrainedBlock, SPORAResidualBlock
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SEED = 99112
LAYERS = (22, 24, 27)
FOUNDATIONS = ROOT / 'data/chinese-foundations.json'
CORRECTIONS = ROOT / 'data/dialogue-correction-training-v1.json'
COMPOSITION = ROOT / 'data/dialogue-composition-v1.json'
REVIEWED = ROOT / 'runs/teaching-curriculum-v5/teacher.jsonl'


def source_dialogues():
    foundations = json.loads(FOUNDATIONS.read_text(encoding='utf-8'))['dialogues']
    correction = json.loads(CORRECTIONS.read_text(encoding='utf-8'))['families']
    reviewed = [json.loads(line) for line in REVIEWED.read_text(encoding='utf-8').splitlines()]
    groups = {'foundations': [row['turns'] for row in foundations], 'correction': [],
              'reviewed': []}
    for rows in correction.values():
        for case in rows:
            groups['correction'].append(list(zip(case[::2], case[1::2], strict=True)))
    for row in reviewed:
        messages = row['messages']
        if len(messages) % 2 or [message['role'] for message in messages] != ['user', 'assistant'] * (len(messages) // 2):
            continue
        texts = [message['content'] for message in messages]
        groups['reviewed'].append(list(zip(texts[::2], texts[1::2], strict=True)))
    return groups


def generated_dialogues():
    """Vary values and which field is requested across matched conversations."""
    plans = ('整理票据', '入库图册', '寄送海报', '清点文具', '标注展板', '归还设备')
    places = ('西侧大厅', '松林站', '东区前台', '石桥路口', '库房三号门', '二楼档案室')
    materials = ('布袋', '胶纸', '纸盒', '砂纸', '卡纸', '胶带')
    separators = ('+', '、', '；', '—')
    rng = random.Random(SEED)
    rows = []
    for _ in range(40):
        action, other = rng.sample(plans, 2)
        left, right = rng.sample(places, 2)
        a, b = rng.sample(materials, 2)
        first, added = rng.sample(range(2, 10), 2)
        divider = rng.choice(separators)
        rows.extend((
            [(f'{action}去{left}，{other}去{right}。请只回{other}的地点。', right),
             (f'现在只把{other}改到{left}，报出新的地点。', left),
             (f'{other}最早说的是哪里？请只发地点。', right)],
            [(f'登记了{first}份{a}，新增{added}份。直接报总数。', str(first + added)),
             (f'新增的那批撤销，报最新还剩几份{a}。', str(first))],
            [(f'填写材料栏，按{b}在前、{a}在后的顺序，用「{divider}」连接。只发结果。',
              divider.join((b, a))),
             (f'现在只留排在后面的名称，直接发文字。', a)],
            [(f'只有地点在{left}时才做{action}，其余地点都做{other}。当前到的是{right}，直接写活动名。', other),
             (f'地点改成{left}，按这个条件填入活动名。', action)],
            [(f'把{a}放入上层，把{b}放入下层。请只发上层的物品名。', a),
             (f'把上层和下层改成放同一种，沿用下层原来的物品。上层现在是什么？', b),
             (f'上层原来放了哪样？只发名称。', a)],
        ))
    return list({tuple(row): row for row in rows}.values())


def tokens(tokenizer, history, question, *, deepseek=True):
    messages = [{'role': role, 'content': text} for user, answer in history
                for role, text in (('user', user), ('assistant', answer))]
    messages.append({'role': 'user', 'content': question})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    suffix = '</think>\n\n' if deepseek else ''
    return tokenizer.encode(prompt + suffix, add_special_tokens=False)


def examples(tokenizer, *, varied, instructor=None, sampled_histories=None, composition=False, dialogue_data=None,
             replace_composition_prefix=False):
    held = {question for _, turns in CASES + FRESH_CASES for question, _ in turns}
    groups = {}
    dialogues_by_group = source_dialogues()
    if varied:
        dialogues_by_group['varied'] = generated_dialogues()
    if composition:
        dialogues_by_group['composition'] = json.loads(COMPOSITION.read_text(encoding='utf-8'))['training']
    if dialogue_data is not None:
        supplemental = json.loads(dialogue_data.read_text(encoding='utf-8'))['training']
        corpus = dialogues_by_group['composition']
        if replace_composition_prefix:
            if (not supplemental or len(supplemental) > len(corpus)
                    or any(len(new) != len(old) for new, old in zip(supplemental, corpus))):
                raise ValueError('The replacement dialogues must preserve the number of turns in each replaced scene')
            corpus[:len(supplemental)] = supplemental
        else:
            corpus.extend(supplemental)
    for group, dialogues in dialogues_by_group.items():
        rows = []
        for conversation in dialogues:
            if any(question in held for question, _ in conversation):
                raise ValueError('Training question occurs in independent evaluation')
            sampled = None
            if group == 'varied' and sampled_histories:
                sampled = sampled_histories.get(tuple(user for user, _ in conversation))
            for index, (question, answer) in enumerate(conversation):
                history = list(conversation[:index])
                if sampled is not None:
                    history = [(user, response) for (user, _), response in zip(history, sampled[:index], strict=True)]
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                if not 2 <= len(target) <= 80:
                    continue
                prompt = tokens(tokenizer, history, question)
                if len(prompt) + len(target) > 512:
                    continue
                echo = tokenizer.encode(question, add_special_tokens=False)
                stale = tokenizer.encode(history[-1][1], add_special_tokens=False) if history else []
                teacher = teacher_request(instructor, history, question, answer) if instructor is not None else None
                rows.append((prompt, target, echo, stale, teacher))
                if history and group == 'correction':
                    corrupted = [(user, user) for user, _ in history]
                    alternate = tokens(tokenizer, corrupted, question)
                    if len(alternate) + len(target) <= 512:
                        teacher = (teacher_request(instructor, corrupted, question, answer)
                                   if instructor is not None else None)
                        rows.append((alternate, target, echo, stale, teacher))
        if replace_composition_prefix and group == 'composition' and len(rows) != sum(map(len, dialogues)):
            raise ValueError('A composition turn exceeded the supported context or answer length')
        if not rows:
            raise ValueError(f'No training requests for {group}')
        groups[group] = rows
    return groups


def collect_histories(checkpoint, output, tokenizer, count):
    if output.exists():
        raise FileExistsError(output)
    digest = sha(checkpoint)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    conversations = random.Random(SEED + 2).sample(generated_dialogues(), count)
    rows = []
    for index, conversation in enumerate(conversations):
        chat = LocalChat(ModelRuntime(runtime.decoder), tokenizer, None, None)
        questions = [question for question, _ in conversation]
        replies = [chat.talk(question) for question in questions[:-1]]
        rows.append({'questions': questions, 'replies': replies})
        if (index + 1) % 10 == 0:
            print('Collected training conversations', index + 1, '/', count, flush=True)
    if sha(checkpoint) != digest:
        raise ValueError('Rollout checkpoint changed')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'histories.json').open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256': digest, 'rows': rows}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def read_histories(path):
    known = {tuple(question for question, _ in conversation) for conversation in generated_dialogues()}
    payload = json.loads(path.read_text(encoding='utf-8'))
    checkpoint_digest = payload['checkpoint_sha256']
    if (not isinstance(checkpoint_digest, str) or len(checkpoint_digest) != 64
            or any(character not in '0123456789abcdef' for character in checkpoint_digest)):
        raise ValueError('Invalid rollout checkpoint digest')
    histories = {}
    for row in payload['rows']:
        key, replies = tuple(row['questions']), row['replies']
        if (key not in known or key in histories or len(replies) != len(key) - 1
                or any(not isinstance(reply, str) or not reply.strip() for reply in replies)):
            raise ValueError('Invalid sampled training history')
        histories[key] = replies
    if not histories:
        raise ValueError('No sampled training histories')
    return histories, checkpoint_digest


def teacher_request(tokenizer, history, question, answer):
    prompt = tokens(tokenizer, history, question, deepseek=False)
    return prompt, tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]


@torch.inference_mode()
def cache_teacher(groups, tokenizer, instructor, directory, digest):
    vocabulary = tokenizer.get_vocab()
    mapping = torch.full((max(instructor.get_vocab().values()) + 1,), -1, dtype=torch.long)
    for token, index in instructor.get_vocab().items():
        if token in vocabulary:
            mapping[index] = vocabulary[token]
    mapping[instructor.eos_token_id] = tokenizer.eos_token_id
    model = load_qwen_streamed(directory, digest)
    mapping = F.pad(mapping, (0, model.config.vocab_size - len(mapping)), value=-1).cuda()
    results = {}
    for group, rows in groups.items():
        aligned = []
        for prefix, target, echo, stale, request in rows:
            prompt, answer = request
            teacher = None
            if len(answer) == len(target) and mapping[answer].tolist() == target and len(prompt) + len(answer) <= 512:
                ids = torch.tensor([prompt + answer[:-1]], device='cuda')
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(answer)).logits[0].float()
                logits[:, mapping < 0] = -torch.inf
                top, indices = logits.topk(32, dim=-1)
                teacher = (mapping[indices].cpu(), (top / 2).softmax(-1).cpu())
            aligned.append((prefix, target, echo, stale, teacher))
        results[group] = aligned
        print('Aligned teacher requests', group, sum(row[-1] is not None for row in aligned), '/', len(rows), flush=True)
    return results


def history_pairs(tokenizer):
    scenarios = (
        ('西侧展厅', '连廊前台', '报备地点先前写的是{old}，现在已经换成{new}。',
         '按修订后的记录，只给出地点。'),
        ('玻璃瓶', '陶杯', '上一稿里的容器是{old}。现在请将它替换成{new}。',
         '只写当前稿采用的容器名称。'),
        ('十一册', '四册', '领取数量原写{old}，请把数量更新成{new}。',
         '请只输出实际应领的数量。'),
        ('蓝卡', '白卡', '桌上有蓝卡和白卡。拿走{old}，留下{new}。',
         '桌上最后剩下哪张卡？只答名称。'),
        ('剪纸', '制陶', '如果只有十分钟就做{old}，时间足够则做{new}。今天时间足够。',
         '活动栏只写一个项目，按前面条件填写。'),
        ('木架', '纸箱', '把{old}放右手边，把{new}放左手边。',
         '现在左手边是什么？只写物品。'),
        ('周一', '周三', '一开始定在{old}，经过调整，定在{new}。',
         '当前日期是哪天？只答星期。'),
        ('易蓉', '马小东', '{old}负责核收报表，{new}负责取快递。',
         '取快递的是谁？请只说全名。'),
        ('外柜', '里柜', '按次序先打开{old}，然后打开{new}。',
         '第二个要打开的是哪个柜？'),
        ('晨间会议', '新书分享', '推送题目原本是{old}，负责人说最终采用{new}。',
         '只回最终采用的题目。'),
        ('清单一', '清单四', '先复核{old}。完成后再核对{new}。',
         '先前安排最后复核哪一份？只报编号。'),
        ('山核桃', '松子', '选购{new}作为新材料，{old}用完就不再购入。',
         '只写本次新采购的材料。'),
    )
    pairs = []
    for left, right, first, followup in scenarios:
        for prior in ('好的。', '看到了。', '', f'那就按{left}处理。'):
            prompts = (tokens(tokenizer, [(first.format(old=left, new=right), prior)], followup),
                       tokens(tokenizer, [(first.format(old=right, new=left), prior)], followup))
            answers = [tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                       for answer in (right, left)]
            pairs.append((prompts, answers))
    return pairs


def forbidden_prefixes(target, phrases):
    negatives = set()
    for phrase in phrases:
        if len(phrase) < 3:
            continue
        for index, token in enumerate(phrase[:len(target)]):
            if token != target[index]:
                negatives.add((index, token))
                break
    return sorted(negatives)


def backpropagate_history_contrast(decoder, prompts, answers):
    total = 0.
    for prompt, preferred, other in zip(prompts, answers, reversed(answers), strict=True):
        index = next(i for i, pair in enumerate(zip(preferred, other)) if pair[0] != pair[1])
        tokens_before = prompt + preferred[:index]
        ids = torch.tensor([tokens_before], device='cuda')
        logits = decoder(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        correct, wrong = logits[preferred[index]], logits[other[index]]
        loss = F.softplus(1.5 + wrong - correct)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite dialogue history loss')
        (.15 * loss).backward()
        total += float(loss.detach())
    return total * .15


def read_trajectory_scores(path, checkpoint_digest, groups):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['checkpoint_sha256'] != checkpoint_digest or not payload['rows']:
        raise ValueError('Trajectory scores are missing or belong to a different checkpoint')
    seen = set()
    for row in payload['rows']:
        index, positions = row['index'], row['positions']
        choices, log_probs = row['choices'], row['log_probs']
        if type(index) is not int or not 0 <= index < len(groups['varied']) or index in seen:
            raise ValueError('Invalid trajectory index')
        seen.add(index)
        prompt, target, *_ = groups['varied'][index]
        if (row['prompt'] != prompt or row['target'] != target or not 1 <= len(row['generated']) <= 12
                or not positions or positions != sorted(set(positions)) or not 0 <= positions[0]
                or positions[-1] >= len(row['generated']) or choices.shape != (len(positions), 32)
                or choices.dtype != torch.long or bool((choices < 0).any()) or bool((choices >= 151936).any())
                or log_probs.shape != choices.shape or not bool(torch.isfinite(log_probs).all())):
            raise ValueError('Invalid trajectory score record')
    return payload['rows']


def backpropagate_trajectory(decoder, row):
    prompt = row['prompt']
    ids = torch.tensor([prompt + row['generated'][:-1]], device='cuda')
    positions = torch.tensor(row['positions'], device='cuda') + len(prompt) - 1
    logits = decoder(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float()
    scores = logits.gather(-1, row['choices'].to(logits.device)) / 2
    student = scores.log_softmax(-1)
    teacher = row['log_probs'].to(logits.device)
    loss = .5 * (student.exp() * (student - teacher)).sum(-1).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite trajectory distillation loss')
    loss.backward()
    return float(loss.detach())


@torch.inference_mode()
def cache_parent_scores(decoder, groups):
    """Keep the parent distribution on replay examples in 32 tokens and a tail bin."""
    scores = {}
    for group in ('foundations', 'correction', 'reviewed'):
        rows = []
        for prompt, target, *_ in groups[group]:
            ids = torch.tensor([prompt + target[:-1]], device='cuda')
            logits = decoder(input_ids=ids, use_cache=False, logits_to_keep=len(target)).logits[0].float()
            values, indices = logits.topk(32, dim=-1)
            probabilities = (values - logits.logsumexp(dim=-1, keepdim=True)).exp()
            tail = (1. - probabilities.sum(dim=-1, keepdim=True)).clamp_min(0.)
            distribution = torch.cat((probabilities, tail), dim=-1)
            rows.append((indices.cpu(), distribution.cpu()))
        scores[group] = rows
        print('Cached parent scores', group, len(rows), flush=True)
    return scores


def replay_loss(logits, indices, teacher):
    student = logits.log_softmax(dim=-1).gather(-1, indices)
    tail = torch.log1p(-student.exp().sum(dim=-1, keepdim=True).clamp(max=1. - 1e-6))
    return F.kl_div(torch.cat((student, tail), dim=-1), teacher, reduction='batchmean')


def fit(decoder, blocks, groups, steps, pairs, loss_tokens=0, trajectories=(), learning_rate=2e-4,
        retention_weight=0.):
    parameters = [weight for block in blocks for weight in (block.down.weight, block.up.weight)]
    parent_scores = cache_parent_scores(decoder, groups) if retention_weight else {}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01)
    rng = random.Random(SEED)
    pair_rng = random.Random(SEED + 1)
    trajectory_rng = random.Random(SEED + 8)
    selected = ('correction', 'foundations', 'reviewed', 'correction', 'foundations')
    if 'varied' in groups:
        selected = ('varied', 'foundations', 'varied', 'correction', 'reviewed')
    if 'composition' in groups:
        selected = ('varied', 'foundations', 'composition', 'correction', 'reviewed')
    trace = []
    running = []
    visits = {}
    projected = 0
    trajectory_exposures = 0
    retention_exposures = 0
    seen = {}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        group = selected[step % len(selected)]
        row_index = rng.randrange(len(groups[group]))
        prompt, target, echo, stale, teacher = groups[group][row_index]
        labels = torch.tensor(target, device='cuda')
        ids = torch.tensor([prompt + target[:-1]], device='cuda')
        positions = list(range(len(target)))
        if loss_tokens and len(target) > loss_tokens:
            row_key = (group, row_index)
            start = visits.get(row_key, 0) * loss_tokens
            positions = sorted((start + offset) % len(target) for offset in range(loss_tokens))
            visits[row_key] = visits.get(row_key, 0) + 1
        offsets = torch.tensor(positions, device='cuda')
        projected += len(positions)
        seen.setdefault((group, row_index), set()).update(positions)
        logits = decoder(input_ids=ids, use_cache=False,
                         logits_to_keep=offsets + len(prompt) - 1).logits[0].float()
        labels = labels.index_select(0, offsets)
        loss = F.cross_entropy(logits, labels)
        if teacher is not None:
            choices, distribution = (tensor[positions].to(logits.device) for tensor in teacher)
            student_logprob = (logits.gather(-1, choices) / 2).log_softmax(-1)
            loss = loss + F.kl_div(student_logprob, distribution, reduction='batchmean')
        if group in parent_scores:
            indices, parent = parent_scores[group][row_index]
            loss = loss + retention_weight * replay_loss(
                logits, indices[positions].to(logits.device), parent[positions].to(logits.device))
            retention_exposures += len(positions)
        negatives = forbidden_prefixes(target, (echo, stale) if group == 'correction' else (echo,))
        selected_rows = {position: i for i, position in enumerate(positions)}
        negatives = [(selected_rows[position], token) for position, token in negatives if position in selected_rows]
        if negatives:
            rows, tokens = zip(*negatives, strict=True)
            probabilities = logits.softmax(-1)[list(rows), list(tokens)]
            loss = loss - .15 * torch.log1p(-probabilities.clamp(max=.9999)).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite dialogue training loss')
        loss.backward()
        reported_loss = float(loss.detach())
        del logits, loss
        if pairs and step % 4 == 3:
            reported_loss += backpropagate_history_contrast(decoder, *pair_rng.choice(pairs))
        if trajectories and step % 4 == 3:
            trajectory = trajectory_rng.choice(trajectories)
            reported_loss += backpropagate_trajectory(decoder, trajectory)
            trajectory_exposures += len(trajectory['positions'])
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        running.append(reported_loss)
        if (step + 1) % 32 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, 'mean_loss': sum(running) / len(running)})
            print(trace[-1], flush=True)
            running.clear()
            torch.cuda.empty_cache()
    exposure = {'requests': len(seen), 'target_positions': sum(map(len, seen.values())),
                'target_exposures': projected,
                'end_targets': sum(len(groups[group][index][1]) - 1 in positions
                                   for (group, index), positions in seen.items())}
    if trajectories:
        exposure['trajectory_exposures'] = trajectory_exposures
    if retention_weight:
        exposure['retention_exposures'] = retention_exposures
    return trace, exposure


def main(checkpoint, output, steps, varied, contrast_history, teacher_model, teacher_cache, prepare_teacher,
         stage_embeddings, loss_tokens, collect_student_histories, student_histories, trained_residuals, composition,
         layers, rank, fresh_holdout, trajectory_scores, dialogue_data=None, continue_residuals=False,
         learning_rate=2e-4, replace_composition_prefix=False, retention_weight=0.):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if collect_student_histories:
        collect_histories(checkpoint, output, tokenizer, collect_student_histories)
        return
    files = [checkpoint, FOUNDATIONS, CORRECTIONS, REVIEWED]
    if composition:
        files.append(COMPOSITION)
    if dialogue_data is not None:
        files.append(dialogue_data)
    if student_histories is not None:
        files.append(student_histories)
    if trajectory_scores is not None:
        files.append(trajectory_scores)
    if teacher_model is not None:
        files.extend(teacher_model / name for name in ('model.safetensors', 'config.json', 'tokenizer_config.json',
                                                       'tokenizer.json', 'generation_config.json'))
    protected = {path: sha(path) for path in files}
    instructor = (None if teacher_model is None else AutoTokenizer.from_pretrained(
        teacher_model, local_files_only=True, trust_remote_code=False))
    histories, rollout_digest = read_histories(student_histories) if student_histories is not None else (None, None)
    groups = examples(tokenizer, varied=varied, instructor=instructor, sampled_histories=histories,
                      composition=composition, dialogue_data=dialogue_data,
                      replace_composition_prefix=replace_composition_prefix)
    trajectories = read_trajectory_scores(trajectory_scores, protected[checkpoint], groups) if trajectory_scores else ()
    print('Training requests', {name: len(rows) for name, rows in groups.items()}, flush=True)
    if instructor is not None:
        provenance = {str(path): digest for path, digest in protected.items()}
        recipes = {group: [(row[0], row[1], row[-1]) for row in rows] for group, rows in groups.items()}
        if prepare_teacher:
            if teacher_cache.exists():
                raise FileExistsError(teacher_cache)
            scored = cache_teacher(groups, tokenizer, instructor, teacher_model,
                                   protected[teacher_model / 'model.safetensors'])
            if any(sha(path) != digest for path, digest in protected.items()):
                raise ValueError('Teacher phase inputs changed')
            teacher_cache.parent.mkdir(parents=True, exist_ok=True)
            atomic_save({'sources': provenance, 'recipes': recipes,
                         'scores': {group: [row[-1] for row in rows] for group, rows in scored.items()}},
                        teacher_cache)
            print('Saved offline teacher scores', teacher_cache, flush=True)
            return
        cached = torch.load(teacher_cache, map_location='cpu', weights_only=True)
        if cached['sources'] != provenance or cached['recipes'] != recipes or cached['scores'].keys() != groups.keys():
            raise ValueError('Offline teacher scores do not match the training inputs')
        groups = {group: [(*row[:-1], score) for row, score in zip(rows, cached['scores'][group], strict=True)]
                  for group, rows in groups.items()}
        protected[teacher_cache] = sha(teacher_cache)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(stage_embeddings=stage_embeddings), MODEL,
                                      disposable_base=True)
    decoder = runtime.decoder.to('cuda').eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    parents = decoder.base.model.layers
    blocks = []
    for index in layers:
        if not 0 <= index < len(parents):
            raise ValueError(f'Layer {index} is outside the decoder')
        layer = parents[index]
        if continue_residuals:
            if (type(layer) is not SPORAResidualBlock or not layer.ordinary_only or layer.rank != rank
                    or layer.memory_context is not decoder._memory_context):
                raise ValueError('Expected an existing ordinary-dialogue residual of the requested rank')
            layer.down.requires_grad_(True)
            layer.up.requires_grad_(True)
            blocks.append(layer)
            continue
        if type(layer) is not SPORAPretrainedBlock or index in decoder.layer_indices:
            raise ValueError('Expected an unmodified compatible layer outside the memory readers')
        block = SPORAResidualBlock(layer, rank=rank, ordinary_only=True,
                                   memory_context=decoder._memory_context).eval()
        parents[index] = block
        blocks.append(block)
    pairs = history_pairs(tokenizer) if contrast_history else []
    recipe = {'sources': {str(path): digest for path, digest in protected.items()},
              'seed': SEED, 'layers': list(layers), 'steps': steps, 'varied': varied,
              'history_pairs': len(pairs), 'loss_tokens_per_step': loss_tokens or None}
    if rank != 16:
        recipe['rank'] = rank
    if continue_residuals:
        recipe['continue_residuals'] = True
    if learning_rate != 2e-4:
        recipe['learning_rate'] = learning_rate
    if replace_composition_prefix:
        recipe['replace_composition_prefix'] = True
    if retention_weight:
        recipe['retention_weight'] = retention_weight
    weights = [part.weight for block in blocks for part in (block.down, block.up)]
    output.mkdir(parents=True, exist_ok=False)
    residual_path = output / 'trained-residuals.pt'
    torch.cuda.empty_cache()
    if trained_residuals is None:
        trace, exposure = fit(decoder, blocks, groups, steps, pairs, loss_tokens=loss_tokens,
                              trajectories=trajectories, learning_rate=learning_rate,
                              retention_weight=retention_weight)
        torch.cuda.empty_cache()
        atomic_save({'recipe': recipe, 'trace': trace, 'exposure': exposure,
                     'weights': [weight.detach().cpu() for weight in weights]}, residual_path)
        print('Saved trained residuals', residual_path, flush=True)
    else:
        saved = torch.load(trained_residuals, map_location='cpu', weights_only=True)
        if saved['recipe'] != recipe or saved['trace'][-1]['step'] != steps or len(saved['weights']) != len(weights):
            raise ValueError('Trained residuals do not match the experiment')
        with torch.no_grad():
            for weight, source in zip(weights, saved['weights'], strict=True):
                if source.shape != weight.shape or source.dtype != weight.dtype or not bool(torch.isfinite(source).all()):
                    raise ValueError('Invalid trained residual weight')
                weight.copy_(source)
        trace = saved['trace']
        exposure = saved.get('exposure')
    decoder.requires_grad_(False)
    if any(sha(path) != digest for path, digest in protected.items()) or frozen_digest(runtime.organism, runtime.state) != state_digest:
        raise ValueError('Experiment changed protected inputs')
    candidate = runtime.save(output / 'integrated.pt')
    print('Saved trained checkpoint', candidate, flush=True)
    dialogue = run(decoder, tokenizer)
    print('Independent dialogue', {key: value for key, value in dialogue.items() if key != 'rows'}, flush=True)
    memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    print('Memory control', memory['summary'], flush=True)
    fresh_dialogue = run(decoder, tokenizer, cases=FRESH_CASES) if histories or composition or fresh_holdout or trajectories else None
    if fresh_dialogue is not None:
        print('Fresh dialogue', {key: value for key, value in fresh_dialogue.items() if key != 'rows'}, flush=True)
    if any(sha(path) != digest for path, digest in protected.items()) or frozen_digest(runtime.organism, runtime.state) != state_digest:
        raise ValueError('Experiment changed protected inputs')
    report = {'candidate_sha256': sha(candidate), **recipe,
              'sampled_history_conversations': len(histories or ()),
              'rollout_checkpoint_sha256': rollout_digest,
              'trace': trace, 'supervision_exposure': exposure,
              'teacher_requests': {name: sum(row[-1] is not None for row in rows) for name, rows in groups.items()},
              'training_requests': {name: len(rows) for name, rows in groups.items()},
              'dialogue': dialogue, 'memory': memory, 'fresh_dialogue': fresh_dialogue}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--continue-residuals', action='store_true', help='Update existing ordinary-dialogue residuals')
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--retention-weight', type=float, default=0.,
                        help='Keep parent predictions on foundation, correction and reviewed replay')
    parser.add_argument('--fresh-holdout', action='store_true')
    parser.add_argument('--varied', action='store_true')
    parser.add_argument('--composition', action='store_true',
                        help='Use the project-authored multiround composition training split')
    parser.add_argument('--dialogue-data', type=Path, help='Add a project dialogue file to the composition group')
    parser.add_argument('--replace-composition-prefix', action='store_true',
                        help='Replace the first composition dialogues with the supplied scenes')
    parser.add_argument('--contrast-history', action='store_true')
    parser.add_argument('--teacher-model', type=Path)
    parser.add_argument('--teacher-cache', type=Path)
    parser.add_argument('--trajectory-scores', type=Path, help='Distill teacher scores on collected student continuations')
    parser.add_argument('--prepare-teacher', action='store_true')
    parser.add_argument('--loss-tokens', type=int, default=0,
                        help='Limit vocabulary projection to rotating answer positions per update')
    parser.add_argument('--stage-embeddings', action='store_true',
                        help='Keep embedding weights on the CPU until decoder assembly finishes')
    history = parser.add_mutually_exclusive_group()
    history.add_argument('--collect-student-histories', type=int, default=0)
    history.add_argument('--student-histories', type=Path)
    parser.add_argument('--trained-residuals', type=Path,
                        help='Package and evaluate saved training weights using the original experiment options')
    args = parser.parse_args()
    if args.steps < 1 or args.loss_tokens < 0:
        parser.error('--steps must be positive and --loss-tokens must be nonnegative')
    if not 0 < args.learning_rate <= .01:
        parser.error('--learning-rate must be in (0, 0.01]')
    if not 0 <= args.retention_weight <= 10 or (args.retention_weight and not args.continue_residuals):
        parser.error('--retention-weight must be in [0, 10] and requires existing residuals when nonzero')
    if args.rank < 1 or min(args.layers) < 0 or list(args.layers) != sorted(set(args.layers)):
        parser.error('Choose a positive rank and distinct layers in increasing order')
    if (args.teacher_model is None) != (args.teacher_cache is None) or (args.prepare_teacher and args.teacher_model is None):
        parser.error('Offline distillation requires both --teacher-model and --teacher-cache')
    if args.collect_student_histories and not 1 <= args.collect_student_histories <= len(generated_dialogues()):
        parser.error('Requested history count exceeds the generated training conversations')
    if args.student_histories is not None and not args.varied:
        parser.error('Student histories require --varied')
    if args.composition and not args.varied:
        parser.error('Composition examples require --varied')
    if args.dialogue_data is not None and not args.composition:
        parser.error('--dialogue-data requires --composition')
    if args.replace_composition_prefix and args.dialogue_data is None:
        parser.error('--replace-composition-prefix requires --dialogue-data')
    if args.trajectory_scores is not None and (not args.varied or args.student_histories or args.collect_student_histories
                                               or args.prepare_teacher or args.teacher_model):
        parser.error('Trajectory scores require the original --varied training histories and answer-only supervision')
    if args.trained_residuals is not None and (args.collect_student_histories or args.prepare_teacher):
        parser.error('Saved training weights require the training or evaluation mode')
    main(**vars(args))
