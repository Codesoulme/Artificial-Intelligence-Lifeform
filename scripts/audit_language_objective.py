"""CPU-only objective audit. Writes only reports/P27-objective-audit.json.

No checkpoint loading, training-entrypoint execution, or held-out text access.
The tiny fit checks plumbing, not Chinese ability. Run with python -B.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
import ast
import collections
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from spora.conversation import ConversationBridge, ConversationConfig, generate, restore_conversation
from spora.conversation_data import DialogueRows
from spora.conversation_optimization import backward_supervised
from spora.growing import new_state
from spora.language_training import interface_state
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.training_mix import choose_source

OUT = ROOT / 'reports/P27-objective-audit.json'
DATA = ROOT / 'runs/conversation-data-v5'
CURRICULUM = ROOT / 'runs/teaching-curriculum-v3'
SOURCE_FILES = [
    'scripts/train_conversation_v7.py', 'spora/conversation.py',
    'spora/conversation_data.py', 'spora/conversation_optimization.py',
    'spora/language_training.py', 'spora/training_mix.py',
    'scripts/prepare_conversation_v2.py', 'scripts/curate_teaching_curriculum.py',
    'configs/p27-coverage-small.json', 'docs/P27-protocol.md',
] + ['scripts/train_conversation' + suffix + '.py'
     for suffix in ('', '_v2', '_v3', '_v4', '_v5', '_v6')]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def training_functions():
    """Execute only named pure helpers, never import or call the GPU trainer."""
    tree = ast.parse((ROOT / SOURCE_FILES[0]).read_text('utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('padded', 'text_batch', 'batch_loss')]
    env = {'torch': torch, 'np': np, 'F': F}
    exec(compile(ast.Module(body=selected, type_ignores=[]), SOURCE_FILES[0], 'exec'), env)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    fraction = next(n.value for n in ast.walk(main) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'fraction' for t in n.targets))
    lr_code = compile(ast.Expression(fraction), SOURCE_FILES[0], 'eval')
    optimizer_nodes = [n for n in main.body if
                       (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets))
                       or (isinstance(n, ast.If) and any(isinstance(x, ast.Call)
                           and isinstance(x.func, ast.Attribute) and x.func.attr == 'load_state_dict'
                           and isinstance(x.func.value, ast.Name) and x.func.value.id == 'optimizer'
                           for x in ast.walk(n)))]
    opt_code = compile(ast.Module(body=optimizer_nodes, type_ignores=[]), SOURCE_FILES[0], 'exec')
    return env, lr_code, opt_code


def reconstruct(tokenizer, messages):
    """Independent raw-message oracle; no production encoder/mask helper."""
    sequence = [tokenizer.token_to_id('<|bos|>')]
    supervised_positions, starts, ends = [], [], []
    roles = []
    for message in messages:
        role = tokenizer.token_to_id('<|' + message['role'] + '|>')
        assert role is not None
        roles.append(len(sequence))
        sequence.append(role)
        first = len(sequence)
        sequence.extend(tokenizer.encode(message['content']).ids)
        sequence.append(tokenizer.token_to_id('<|eos|>'))
        if message['role'] == 'assistant':
            starts.append(first)
            ends.append(len(sequence) - 1)
            supervised_positions.extend(range(first, len(sequence)))
    return sequence, supervised_positions, starts, ends, roles


def sample_raw(name, total, count):
    indices = {round(i * (total - 1) / (count - 1)) for i in range(count)}
    result = []
    with (CURRICULUM / (name + '.jsonl')).open('rb') as raw, \
            (CURRICULUM / (name + '-tokens.jsonl')).open('rb') as encoded:
        for i, (a, b) in enumerate(zip(raw, encoded)):
            if i in indices:
                result.append((i + 1, json.loads(a), json.loads(b)))
    return result


def audit_records(tokenizer, padded):
    manifest = read_json(CURRICULUM / 'manifest.json')
    groups = {
        'teacher': sample_raw('teacher', manifest['stats']['teacher_dialogues'], 16),
        'foundations': sample_raw('foundations', manifest['foundations'], 12),
    }
    # Small legacy dialogue sample. Filtering mirrors file preparation's length
    # policy only; expected labels below come directly from raw role spans.
    selected = []
    with (DATA / 'dialogue-train.jsonl').open('rb') as raw, \
            (DATA / 'dialogue-train-tokens.jsonl').open('rb') as encoded:
        for i, line in enumerate(raw):
            row = json.loads(line)
            seq, *_ = reconstruct(tokenizer, row['messages'])
            if len(seq) <= 512:
                selected.append((i + 1, row, json.loads(next(encoded))))
            if len(selected) == 8:
                break
    groups['legacy_dialogue_train'] = selected
    output, samples = {}, []
    for name, records in groups.items():
        counter = collections.Counter()
        evidence = []
        for line, raw, row in records:
            seq, targets, firsts, ends, roles = reconstruct(tokenizer, raw['messages'])
            assert raw['group'] == row['group']
            assert row['tokens'] == seq
            assert [i for i, valid in enumerate(row['mask']) if valid] == targets
            length = min(512, 32 * math.ceil((len(seq) - 1) / 32))
            x, y = padded([row], length, 0, 'cpu')
            expected = torch.full_like(y, -100)
            for j in targets:
                expected[0, j - 1] = seq[j]
                assert x[0, :j].tolist() == seq[:j]
            assert torch.equal(expected, y)
            assert all(y[0, j - 1].item() == seq[j] for j in firsts + ends)
            assert all(y[0, j - 1].item() == -100 for j in roles)
            counter.update(records=1, targets=len(targets), assistant_starts=len(firsts),
                           assistant_eos=len(ends), content_characters=sum(len(m['content']) for m in raw['messages']),
                           content_tokens=sum(len(tokenizer.encode(m['content']).ids) for m in raw['messages']))
            for m in raw['messages']:
                assert tokenizer.decode(tokenizer.encode(m['content']).ids) == m['content']
            evidence.append({'line': line, 'group': row['group'], 'tokens': len(seq),
                             'targets': len(targets), 'first_target_positions_in_labels': [j-1 for j in firsts],
                             'eos_target_positions_in_labels': [j-1 for j in ends]})
            if name == 'foundations':
                samples.append((raw, row))
        output[name] = {'counts': dict(counter), 'evidence': evidence, 'all_alignment_checks_passed': True}
    focus = read_json(DATA / 'focus-tokens.json')
    assert all(seq[0] == 1 and seq[-1] == 2 and len(seq) <= 33 for seq in focus)
    assert all(not any(t in (0, 3, 4) for t in seq) for seq in focus)
    x, y = padded([{'tokens': seq, 'mask': [True] * len(seq)} for seq in focus], 32, 0, 'cpu')
    assert sum(int((r != -100).sum()) for r in y) == sum(len(seq)-1 for seq in focus)
    output['focus'] = {'segments': len(focus), 'min_length': min(map(len, focus)),
                       'max_length': max(map(len, focus)), 'all_eos_supervised': True,
                       'no_role_tokens': True, 'fits_smallest_rounded_training_length': True}
    # Read only the first three TRAIN raw text records and corresponding bytes.
    with (DATA / 'pretrain-train.jsonl').open('rb') as stream:
        raw_text = [json.loads(line)['text'] for line in itertools.islice(stream, 3)]
    seq = list(itertools.chain.from_iterable([1] + tokenizer.encode(text).ids + [2] for text in raw_text))
    stream = np.memmap(DATA / 'pretrain-train.bin', dtype=np.uint16, mode='r', shape=(len(seq),))
    assert stream.tolist() == seq
    output['pretrain_prefix'] = {'raw_train_documents': 3, 'matched_tokens': len(seq), 'uint16_encoding_matches': True}
    del stream
    return output, samples


def replay_mix(plan, lr_code):
    teacher = DialogueRows(CURRICULUM / 'teacher-tokens.jsonl', plan['length'])
    foundation = DialogueRows(CURRICULUM / 'foundations-tokens.jsonl', plan['length'])
    pools = {'teacher': teacher, 'foundation': foundation}
    focus = read_json(DATA / 'focus-tokens.json')
    stream_size = (DATA / 'pretrain-train.bin').stat().st_size // 2
    rng = random.Random(plan['seed'])
    counts, coefficients, lr_coefficients = {}, collections.Counter(), collections.Counter()
    cursor = 0
    summaries = collections.defaultdict(list)
    try:
        for step in range(1, 501):
            kind = choose_source(plan['target_shares'], counts)
            if kind == 'pretrain':
                for _ in range(plan['batch']):
                    rng.randrange(stream_size - plan['length'])
                target_counts = [plan['length']] * plan['batch']
            else:
                rows = [rng.choice(pools[kind]) for _ in range(plan['batch'])]
                target_counts = [sum(r['mask'][1:]) for r in rows]
            f = 0
            if step % plan['focus_every'] == 0:
                f = len(focus[cursor % len(focus)]) - 1
                cursor += 1
                target_counts[0] = f
                counts['focus_targets'] = counts.get('focus_targets', 0) + f
            n = sum(target_counts)
            for suffix, amount in (('_steps', 1), ('_targets_including_focus', n), ('_source_targets', n-f)):
                counts[kind + suffix] = counts.get(kind + suffix, 0) + amount
            lr_fraction = eval(lr_code, {'math': math, 'min': min, 'step': step, 'plan': plan})
            coefficients[kind] += (n-f)/n
            coefficients['focus'] += f/n
            lr_coefficients[kind] += lr_fraction*(n-f)/n
            lr_coefficients['focus'] += lr_fraction*f/n
            summaries[kind].append(n)
    finally:
        teacher.close(); foundation.close()
    recorded = ROOT / 'runs/p27-coverage-small/segment-000500.json'
    matches = counts == read_json(recorded)['counts'] if recorded.exists() else None
    total = sum(counts.get(k + '_source_targets', 0) for k in plan['target_shares'])
    return {
        'steps_replayed_without_model': 500, 'counts': counts, 'matches_recorded_segment_counts': matches,
        'source_token_shares': {k: counts[k+'_source_targets']/total for k in plan['target_shares']},
        'mean_loss_coefficient_shares': {k: v/500 for k, v in coefficients.items()},
        'lr_weighted_mean_loss_coefficient_shares': {k: v/sum(lr_coefficients.values()) for k, v in lr_coefficients.items()},
        'targets_per_update': {k: {'min': min(v), 'max': max(v), 'mean': sum(v)/len(v)} for k, v in summaries.items()},
        'interpretation': 'These are coefficients of source token-mean losses across updates, not measured gradient magnitudes. AdamW and clipping prevent equating them with exact influence or parameter movement.'
    }


def tiny_model(tokenizer):
    torch.manual_seed(270811)
    organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
        expert_rank=4, memory_layers=(0, 1), history_writer='rls', current_address='orthogonal'))
    state = new_state(organism)
    with torch.no_grad():
        for matrix in state.current:
            matrix.normal_(0, .03)
    config = ConversationConfig(vocabulary=tokenizer.get_vocab_size(), width=32, depth=2,
        heads=4, hidden=64, context=96, memory_layers=(0, 1), memory_read='legacy')
    model = ConversationBridge(organism, config)
    payload = {'format': 'spora-conversation-v1', 'organism_sha256': 'toy-parent',
               'tokenizer_sha256': 'toy-tokenizer', 'config': asdict(config), 'state': interface_state(model)}
    model = restore_conversation(organism, payload, 'toy-parent', 'toy-tokenizer', 'cpu')
    return model, state


def tiny_fit(tokenizer, helpers, opt_code, plan):
    model, state = tiny_model(tokenizer)
    # Fresh, explicitly synthetic toy strings, not development/final questions.
    messages = [
        [{'role': 'user', 'content': '只说山。'}, {'role': 'assistant', 'content': '山。'}],
        [{'role': 'user', 'content': '只说水。'}, {'role': 'assistant', 'content': '水。'}],
        [{'role': 'user', 'content': '只说风。'}, {'role': 'assistant', 'content': '风。'},
         {'role': 'user', 'content': '再说雨。'}, {'role': 'assistant', 'content': '雨。'}],
    ]
    rows, generation = [], []
    for dialogue in messages:
        seq, targets, firsts, ends, _ = reconstruct(tokenizer, dialogue)
        rows.append({'tokens': seq, 'mask': [i in targets for i in range(len(seq))]})
        for first, end in zip(firsts, ends):
            generation.append((seq[:first], seq[first:end], 'assistant'))
    for text in ('青山。绿水。', '微风。细雨。'):
        content = tokenizer.encode(text).ids
        seq = [1] + content + [2]
        rows.append({'tokens': seq, 'mask': [False] + [True]*(len(seq)-1)})
        generation.append((seq[:2], seq[2:-1], 'plain_continuation'))
    x, y = helpers['padded'](rows, 32, 0, 'cpu')
    assert x.shape == y.shape
    # v7's actual optimizer constructor/resume statements, isolated from main.
    env = {'torch': torch, 'model': model, 'plan': plan, 'payload': None}
    exec(opt_code, env)
    production_opt = env['optimizer']
    reset_empty = len(production_opt.state) == 0
    named = dict(model.named_parameters())
    initial = {n: p.detach().clone() for n, p in named.items()}
    target_counts = [int((row != -100).sum()) for row in y]
    with torch.no_grad():
        initial_loss = float(helpers['batch_loss'](model, state, x, y))
    production_opt.zero_grad(set_to_none=True)
    first_loss = backward_supervised(model, state, x, y, 2)
    gradient = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in named.items()}
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
    production_opt.step()
    delta = {n: float((p.detach()-initial[n]).abs().max()) for n, p in named.items()}
    # Validate optimizer loading on small in-memory state; no saved weights read.
    opt_state = copy.deepcopy(production_opt.state_dict())
    resumed_env = {'torch': torch, 'model': model, 'plan': plan, 'payload': {'optimizer': opt_state}}
    exec(opt_code, resumed_env)
    resumed = resumed_env['optimizer']
    resume_steps = sorted({int(s['step']) for s in resumed.state.values()})
    # Deliberately larger toy LR to test learnability in <=100 steps. This is not
    # a test of the production LR's adequacy for Chinese or for large checkpoints.
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.01, weight_decay=.01, betas=(.9, .95))
    trace = []
    for step in range(1, 80):
        optimizer.zero_grad(set_to_none=True)
        loss = backward_supervised(model, state, x, y, 2)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (1, 19, 39, 59, 79):
            trace.append({'toy_update': step + 1, 'pre_update_loss': float(loss)})
    with torch.no_grad():
        logits = model(x, state)
        token_losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='none').reshape_as(y)
        final_loss = float(token_losses.sum()/(y != -100).sum())
        # Explicit prefix-by-prefix likelihood at assistant starts/EOS validates
        # the same target locations used by autoregressive inference.
        prefix_evidence = []
        for row_index, row in enumerate(rows[:3]):
            seq = row['tokens']
            for j in range(1, len(seq)):
                if row['mask'][j] and (seq[j-1] == 4 or seq[j] == 2):
                    scores = model(torch.tensor([seq[:j]]), state)[0, -1]
                    expected = seq[j]
                    prefix_loss = float(F.cross_entropy(scores[None], torch.tensor([expected])))
                    prefix_evidence.append({'row': row_index, 'target_position': j,
                        'kind': 'eos' if expected == 2 else 'assistant_first',
                        'target': expected, 'argmax': int(scores.argmax()),
                        'prefix_loss': prefix_loss, 'batched_loss': float(token_losses[row_index, j-1])})
    generated = []
    for prefix, expected, kind in generation:
        actual = generate(model, state, prefix, 2, budget=16, forbidden=(0, 1, 3, 4))
        generated.append({'kind': kind, 'exact': actual['tokens'] == expected and actual['ended_with_eos'],
                          'expected': tokenizer.decode(expected), 'actual': tokenizer.decode(actual['tokens']),
                          'ended_with_eos': actual['ended_with_eos']})
    language = [n for n in named if not n.startswith('organism.')]
    frozen = [n for n in named if n.startswith('organism.')]
    return {
        'architecture': asdict(model.config), 'language_parameters': sum(named[n].numel() for n in language),
        'updates': 80, 'first_update_lr': plan['learning_rate'], 'remaining_79_updates_lr': .01,
        'schedule_scope': 'The update probe uses the production base LR, not the first warmup LR; the production schedule is audited separately without training.',
        'model_training_flag': model.training, 'mode_explanation': 'restore returns eval; this network has no dropout/batchnorm, so eval does not disable autograd.',
        'initial_loss': initial_loss, 'final_loss': final_loss, 'trace': trace,
        'first_microbatch_loss_matches_full': abs(float(first_loss)-initial_loss) < 2e-6,
        'unequal_supervised_counts': target_counts,
        'language_tensors': len(language), 'language_tensors_with_nonzero_gradient': sum(bool(gradient[n]) for n in language),
        'language_tensors_changed_after_production_optimizer_step': sum(delta[n] > 0 for n in language),
        'per_tensor_first_step': {n: {'gradient_norm': gradient[n], 'max_abs_update': delta[n]} for n in language},
        'frozen_parameter_tensors': len(frozen),
        'frozen_parameters_unchanged': all(torch.equal(initial[n], named[n].detach()) for n in frozen),
        'frozen_parameters_have_no_grad': all(named[n].grad is None for n in frozen),
        'new_fork_optimizer_empty': reset_empty, 'resumed_optimizer_steps': resume_steps,
        'resumed_optimizer_state_tensors': len(resumed.state),
        'first_and_eos_prefix_evidence': prefix_evidence,
        'max_prefix_batched_loss_difference': max(abs(e['prefix_loss']-e['batched_loss']) for e in prefix_evidence),
        'toy_generation': generated,
        'limits': 'A small random model memorizing six prompted continuations does not establish Chinese competence or prove that live large checkpoints updated. No pretrained weights were loaded.'
    }


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    before = {name: digest(ROOT/name) for name in SOURCE_FILES}
    helpers, lr_code, opt_code = training_functions()
    plan = read_json(ROOT/'configs/p27-coverage-small.json')
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizer.json'))
    tokenizer_sha = digest(DATA/'tokenizer.json')
    assert tokenizer_sha == read_json(DATA/'manifest.json')['tokenizer_sha256']
    assert tokenizer_sha == read_json(CURRICULUM/'manifest.json')['tokenizer_sha256']
    print('Audit: raw-message target alignment', flush=True)
    records, _ = audit_records(tokenizer, helpers['padded'])
    print('Audit: deterministic source replay (no model)', flush=True)
    mix = replay_mix(plan, lr_code)
    print('Audit: 80-update tiny CPU fit and parameter accounting', flush=True)
    fit = tiny_fit(tokenizer, helpers, opt_code, plan)
    lr = {str(step): plan['learning_rate']*eval(lr_code, {'math': math, 'min': min, 'step': step, 'plan': plan})
          for step in (1, 100, 101, 500, 2000, 20000)}
    after = {name: digest(ROOT/name) for name in SOURCE_FILES}
    assert before == after, 'Shared source changed during audit; do not certify a moving snapshot'
    helper_variants = {}
    for filename in SOURCE_FILES:
        if not Path(filename).name.startswith('train_conversation'):
            continue
        tree = ast.parse((ROOT/filename).read_text('utf-8-sig'))
        helper_variants[filename] = ast.dump(next(n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == 'padded'), include_attributes=False)
    assert len(set(helper_variants.values())) == 1
    report = {
        'schema_version': 1, 'date': '2026-11-08', 'phase': 'P23/P27 language objective independent audit',
        'constraints': {'cpu_threads': torch.get_num_threads(), 'cpu_interop_threads': torch.get_num_interop_threads(),
            'gpu_used': False, 'production_weights_loaded': False, 'final_test_text_opened': False,
            'shared_implementation_modified': False, 'existing_unit_suite_rerun': False,
            'writes': ['scripts/audit_language_objective.py', 'reports/P27-objective-audit.json']},
        'source_sha256': before, 'shared_sources_unchanged': True,
        'script_sha256': digest(Path(__file__)), 'plan': plan,
        'tokenizer_sha256': tokenizer_sha, 'tokenizer_matches_both_manifests': True,
        'historical_padded_helpers_identical': list(helper_variants),
        'sample_alignment': records, 'source_mix_replay': mix, 'tiny_fit': fit,
        'production_lr_schedule': lr,
        'verdict': {
            'confirmed_implementation_bugs': 0,
            'root_cause_of_weak_production_language_identified': False,
            'summary': '未发现系统性错位、mask/角色/EOS错误或语言主体被整体冻结的实现缺陷。当前证据排除所测路径上的这些候选原因；无法在不读取实际权重的条件下证明线上参数确已更新。唯一具体的优化目标风险是跨批次按目标数调度、却按每批均值更新，两者并不等于同一词元加权总体目标。',
            'do_not_modify_running_training': True,
        },
        'findings': [{
            'key': 'P27-O1', 'priority': 'P2', 'classification': 'training_recipe_objective_risk_not_confirmed_implementation_bug',
            'title': '40/59/1词元曝光不能解读为40/59/1总体损失权重',
            'locations': ['scripts/train_conversation_v7.py:149-168', 'spora/conversation_optimization.py:14-24', 'spora/training_mix.py:10-15'],
            'evidence': '前500步重放与已记录counts完全相等：普通文本56次、教学422次、小教材22次。每次更新都按自身有效目标数取平均，短批次的单个目标因而获得更大的损失系数。',
            'coefficient_shares': mix['mean_loss_coefficient_shares'],
            'why_it_matters': '若用40%普通文本来推断优化器对基础续写的投入，会高估相对权重；小教材的重复拟合也可能明显强于1%的曝光比例所暗示的权重。此差异能影响语言基础/指令学习的平衡，但没有生产权重对照，不能认定它已解释实际弱表现。',
            'not_a_microbatch_bug': '同一批内部的sum/total梯度累计正确；差异发生在长度不同的多个完整更新之间。',
            'protocol_assessment': '协议承诺的是有效词元分配，现实现确实兑现该计数；若目标只是曝光占比，则是已量化的配方性质，而非违反协议。只有将其当成总体词元损失的混合权重时才存在目标错配。',
            'action': '不改正在训练的冻结源码。后续另立配方对照，先明确优化目标究竟是曝光份额还是跨来源的目标词元平均损失；若选择后者，可比较固定有效目标预算的更新或固定分母累计，再独立评估普通续写与聊天。单纯缩放每步loss在AdamW与裁剪下不保证等价，须单独验证。',
            'reproduction': '.venv/Scripts/python.exe -B scripts/audit_language_objective.py；检查source_mix_replay。重放不加载模型或最终题。',
        }],
        'static_assessments': {
            'alignment': 'padded把seq[j]且mask[j]=true放在labels[j-1]，对应输入seq[:j]。mask不是按输入位置解释。',
            'roles_and_eos': '角色标记由调用方提供并不作为生成目标；assistant内容首项及其EOS受监督，user内容和user EOS不受对话损失监督。EOS复用作轮次/文档结束符合生成器停止约定。',
            'padding': '仅右侧填充，因果注意力阻止填充影响左侧有效词元；无须为此额外遮掉历史有效位置。DialogueRows只保留完整且有目标的行；实际样本没有截断答案。',
            'microbatch': '每个子批CE使用reduction=sum再除整个批的有效目标数，末尾不足一个子批和空目标子批不会改变目标分母。现有梯度等价单测已阅读，不再镜像重复。',
            'freeze': 'ConversationBridge只冻结organism；embedding、全部语言blocks、norm、memory_query/out/gate保持requires_grad。共享输入输出embedding仍是可训练参数。',
            'eval_mode': 'restore返回eval且训练器未切train，但当前语言路径没有dropout/batchnorm；eval不等同于no_grad，不会阻止更新。若将来加入依赖模式的层才需另行处理。',
            'optimizer_policy': '新课程分支payload=None，继承语言张量后新建AdamW；resume分支从payload恢复optimizer、step、随机数和计数。reset不是每批执行。符合P27及P23后续课程所声明的显式重置。',
            'schedule': 'step是分支内累计步数，不随每个500步段归零；首次热身1e-6，第100步1e-4，20000步1e-5。祖先步数用于核对而非冒充本分支调度步数。',
            'focus': '32段长度12至25，最小32长度批也容得下完整EOS；只替换一条样本，标签一致，计数单独记账。前500步仅781个focus目标，未见其耗尽主体监督。',
            'data_and_capacity': '可用新预训练文件大小不等于已学习量：首500步仅456704个普通文本目标，对文件246087365词元约0.186%的有放回曝光量。教学行可重复出现，已有开发小教材NLL很低也不代表未见中文能力。容量不足、知识质量及泛化问题均未被本诊断证明或排除。',
        },
        'follow_up_without_overclaim': [
            '保留160项回归基线；无需因本报告修改正在运行的实现。',
            '以后在允许检查训练状态时，比较相邻生产检查点各语言层参数差值及AdamW step，才能直接核实真实更新；本报告不代替该证据。',
            '任何配方变更应单独建运行并保持对照，不把玩具拟合或训练损失下降作为中文能力验收。',
        ],
        'hypotheses': [
            {'hypothesis': 'Shift/mask drops assistant first token or EOS', 'result': 'Not reproduced in independent raw-to-label checks.'},
            {'hypothesis': 'Microbatch gives short and long rows equal weight', 'result': 'Implementation sums CE and divides by full-batch valid-token count; scalar loss agrees on real tokenizer toy rows. Existing gradient-equivalence unit test was read, not duplicated.'},
            {'hypothesis': 'Only frozen organism exists in the optimizer', 'result': 'Language stack, tied embedding and read adapters all receive gradients and change in the tiny restored model; organism remains frozen.'},
            {'hypothesis': 'Source token allocation equals the effective objective mixture', 'result': 'Not equivalent: homogeneous batches have different target counts but each update uses its own token mean. This is a recipe interpretation risk, not an off-by-one defect.'},
        ],
        'elapsed_seconds': time.perf_counter()-started,
        'limitations': [
            'No production checkpoint tensors or optimizer states loaded; actual live weight deltas are unverified.',
            'Only sampled raw TRAIN rows were semantically decoded. Alignment is not a factual-quality audit.',
            'No final/evaluation test text opened. Directory filenames and manifest counts/hashes were visible.',
            'Teacher tokens were scanned to build the same DialogueRows index; raw content was decoded only for16 teacher,12 foundation and8 accepted legacy-dialogue records (plus any skipped overlength prefix rows).',
            'Only3 pretraining raw records and their leading binary tokens were compared; no full pretraining quality assessment.',
            '80 toy updates use small width/depth and accelerated LR for79 steps. Memorization is not language capability or proof that the production recipe is sufficient.',
            'Source-loss coefficients are not measured gradient contributions; adaptive optimization, clipping and sequential updates matter.',
            'The original160-test passing baseline is user-reported; no regression suite was repeated.',
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT), 'initial_loss': fit['initial_loss'], 'final_loss': fit['final_loss'],
        'generated_exact': sum(r['exact'] for r in fit['toy_generation']),
        'generated_total': len(fit['toy_generation']), 'mix_matches_segment': mix['matches_recorded_segment_counts'],
        'language_changed': fit['language_tensors_changed_after_production_optimizer_step'],
        'language_tensors': fit['language_tensors'], 'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
