"""Controlled multi-relation writes and corrections with old-fact replay.

Labels are supplied by this experiment. This does not implement observation,
autonomous writing, historical retrieval, or a deployed individual.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_adapter_restart_audit as binding_source
from scripts import deepseek_spora_correction_generalization_audit as prior
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_resume_equivalence_audit as resume
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-multirelation-correction-20261116'
STEPS = 84
TEMPLATES = {
    'drink': {'train': data.TRAIN_TEMPLATES,
              'held': (data.HELD_TEMPLATES[0], '谈到饮品，{name}喜欢喝哪一种？')},
    'city': {'train': ('请只回答{name}住在哪个城市。', '记录中{name}居住的城市是哪里？'),
             'held': ('{name}目前居住在哪里？', '说出{name}的居住城市。')},
    'hobby': {'train': ('请只回答{name}喜欢什么爱好。', '记录中{name}的爱好是什么？'),
              'held': ('{name}平时喜欢做什么？', '说出{name}的个人爱好。')},
}
ORIGINAL = {('小林', 'drink'): '豆浆', ('阿宁', 'drink'): '红茶', ('小雨', 'drink'): '果汁'}
NEW = {('小林', 'city'): '宜昌', ('阿岚', 'city'): '汕头',
       ('小乔', 'hobby'): '园艺', ('阿宁', 'hobby'): '陶艺'}
LEARNED = {**ORIGINAL, **NEW}
CORRECTED = {**LEARNED, ('小林', 'city'): '乐山', ('小乔', 'hobby'): '摄影'}
UNKNOWN_TRAIN = (('阿哲', 'drink'), ('阿哲', 'city'), ('阿哲', 'hobby'),
                 ('小林', 'hobby'), ('阿宁', 'city'), ('阿岚', 'hobby'), ('小乔', 'drink'))
UNKNOWN_HELD = tuple((name, relation) for name in ('小魏', '阿敏') for relation in TEMPLATES)
MISSING_KNOWN = (('小林', 'hobby'), ('阿宁', 'city'), ('阿岚', 'hobby'), ('小乔', 'drink'))
AUDIT_NAMES = ('阿遥', '小芮', '阿卓', '小彤', '阿骏', '小澄', '阿薇', '小梁')


def validate_design():
    if set(UNKNOWN_TRAIN + UNKNOWN_HELD + MISSING_KNOWN) & set(LEARNED):
        raise ValueError('Unknown and known pairs must not overlap')
    if set(CORRECTED) != set(LEARNED) or len(LEARNED) != 7:
        raise ValueError('Invalid fact or correction set')
    for relation in TEMPLATES:
        if set(TEMPLATES[relation]['train']) & set(TEMPLATES[relation]['held']):
            raise ValueError('Training and held templates overlap')


def examples(facts):
    cases = []
    for (name, relation), value in facts.items():
        subgroup = 'old' if (name, relation) in ORIGINAL else 'new'
        for split in ('train', 'held'):
            for template in TEMPLATES[relation][split]:
                cases.append({'group': f'{subgroup}_{split}', 'name': name, 'relation': relation,
                              'question': template.format(name=name), 'expected': value})
    for group, pairs in (('unknown_person', UNKNOWN_HELD), ('unknown_relation', MISSING_KNOWN)):
        for name, relation in pairs:
            for template in TEMPLATES[relation]['held']:
                cases.append({'group': group, 'name': name, 'relation': relation,
                              'question': template.format(name=name), 'expected': '我不知道'})
    return cases


def main():
    validate_design()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous_result = json.loads((prior.OUT / 'result.json').read_text(encoding='utf-8'))
    original_binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    source = prior.OUT / 'stage-60.pt'
    source_sha = base.sha(source)
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if (source_sha != previous_result['stages'][-1]['candidate_sha256']
            or base_sha != base.WEIGHT_SHA or base_sha != original_binding['base_sha256']
            or feature_sha != original_binding['feature_sha256'] or files != original_binding['files']):
        raise ValueError('Source candidate or model binding mismatch')
    train_controls, held_controls = data.load_reviewed_prompts()
    control_prompts = [row['prompt'] for row in train_controls]
    ordinary = [('ordinary_train', item['prompt']) for item in train_controls]
    ordinary += [('ordinary_held', item['prompt']) for item in held_controls]
    old_rows = {row['question']: row for row in previous_result['stages'][-1]['rows']}
    base_results = json.loads((binding_source.OUT / 'training.json').read_text(encoding='utf-8'))
    base_rows = {row['question']: row['no_adapter'] for row in base_results['states'][-1]['rows']}
    torch.manual_seed(data.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in binding_source.LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    parameters = [parameter for module in modules for part in (module.down, module.up)
                  for parameter in part.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0, foreach=False)
    info = checkpoint.restore(source, modules, base_sha256=base_sha, feature_sha256=feature_sha,
                              layers=binding_source.LAYERS, optimizer=optimizer)

    def prefix(question):
        return pilot.closed_think_inputs(tokenizer, question, 'cuda')['input_ids']

    def answer_loss(question, answer):
        inputs = prefix(question)
        tokens = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor([tokens], dtype=torch.long, device='cuda')
        full = torch.cat((inputs, target[:, :-1]), dim=1)
        scores = model(input_ids=full, use_cache=False, return_dict=True).logits
        scores = scores[:, inputs.shape[1] - 1:inputs.shape[1] - 1 + target.shape[1]].float()
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))

    def first_logits(inputs):
        return model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()

    controls = [prefix(question) for question in control_prompts]
    with torch.no_grad():
        teacher = [F.softmax(first_logits(item), -1).detach() for item in controls]

    all_results, losses = [], []
    for phase, facts in (('before', ORIGINAL), ('learned', LEARNED), ('corrected', CORRECTED)):
        if phase != 'before':
            fact_items = tuple(facts.items())
            for step in range(STEPS):
                index = (step // 2) % len(fact_items)
                (name, relation), value = fact_items[index]
                template_index = step % 2
                question = TEMPLATES[relation]['train'][template_index].format(name=name)
                unknown_name, unknown_relation = UNKNOWN_TRAIN[(step // 2 + step // 14) % len(UNKNOWN_TRAIN)]
                unknown_question = TEMPLATES[unknown_relation]['train'][template_index].format(name=unknown_name)
                optimizer.zero_grad(set_to_none=True)
                fact_loss = answer_loss(question, value)
                fact_loss.backward()
                unknown_loss = answer_loss(unknown_question, '我不知道。')
                (0.5 * unknown_loss).backward()
                control_index = (step // 2) % len(controls)
                kl = F.kl_div(F.log_softmax(first_logits(controls[control_index]), -1),
                              teacher[control_index], reduction='batchmean')
                (0.5 * kl).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append({'phase': phase, 'step': step + 1, 'fact': float(fact_loss.detach()),
                               'unknown': float(unknown_loss.detach()), 'kl': float(kl.detach())})
                if (step + 1) % 28 == 0:
                    print(json.dumps({'phase': phase, 'step': step + 1, 'fact': losses[-1]['fact']}), flush=True)
        questions = examples(facts)
        if phase == 'before':
            for case in examples(LEARNED):
                if case['group'].startswith('new_'):
                    case['group'] = 'new_before_' + case['group'].split('_')[1]
                    case['expected'] = '我不知道'
                    questions.append(case)
        questions += [{'group': group, 'question': question} for group, question in ordinary]
        rows = []
        for case in questions:
            answer = pilot.generate(model, tokenizer, case['question'])
            row = {**case, 'answer': answer}
            if 'expected' in case:
                row['exact'] = prior.score(answer['answer'], case['expected'])
            if case['group'].startswith('ordinary'):
                reference = old_rows[case['question']]
                row['same_start'] = answer['token_ids'] == reference['answer']['token_ids']
                row['same_base'] = answer['token_ids'] == base_rows[case['question']]['token_ids']
            rows.append(row)
        candidate = OUT / f'{phase}.pt'
        phase_steps = {'before': 0, 'learned': STEPS, 'corrected': 2 * STEPS}[phase]
        checkpoint.save(candidate, modules, optimizer, base_sha256=base_sha, feature_sha256=feature_sha,
                        layers=binding_source.LAYERS, steps=info['steps'] + phase_steps)
        all_results.append({'phase': phase, 'rows': rows, 'candidate_sha256': base.sha(candidate)})
        print(json.dumps({'phase': phase, 'evaluated': len(rows)}), flush=True)
    if base.sha(source) != source_sha or base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Source model or candidate changed')
    result = {'source_sha256': source_sha, 'base_sha256': base_sha, 'steps_per_phase': STEPS,
              'phases': all_results, 'losses': losses, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Controlled labeled facts, not autonomous observation.',
                              'Full replay of all current facts in each phase.',
                              'Synthetic development prompts and names; no independent benchmark.',
                              'First-token KL is not a full chat preservation guarantee.',
                              'Short-answer exact matching; generation capped at 24 tokens.']}
    resume.write_json(OUT / 'result.json', result)
    print(json.dumps({'seconds': result['seconds']}), flush=True)


def audit_unknown_people():
    validate_design()
    if set(AUDIT_NAMES) & {name for name, _ in LEARNED}:
        raise ValueError('Unknown audit name overlaps facts')
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    candidate = OUT / 'corrected.pt'
    feature_sha, files = binding_source.binding()
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    if (base_sha != base.WEIGHT_SHA or base_sha != result['base_sha256']
            or feature_sha != binding['feature_sha256'] or files != binding['files']
            or base.sha(candidate) != result['phases'][-1]['candidate_sha256']):
        raise ValueError('Candidate or model binding mismatch')
    torch.manual_seed(data.SEED)
    torch.set_num_threads(2)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in binding_source.LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    checkpoint.restore(candidate, modules, base_sha256=base_sha, feature_sha256=feature_sha,
                       layers=binding_source.LAYERS)
    rows = []
    for name in AUDIT_NAMES:
        for relation, templates in TEMPLATES.items():
            for template in templates['held']:
                question = template.format(name=name)
                answer = pilot.generate(model, tokenizer, question)
                rows.append({'name': name, 'relation': relation, 'question': question, 'answer': answer,
                             'strict_unknown': prior.score(answer['answer'], '我不知道')})
        print(json.dumps({'person': name, 'completed': len(rows)}, ensure_ascii=False), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base weights changed')
    report = {'candidate_sha256': result['phases'][-1]['candidate_sha256'], 'rows': rows,
              'strict_unknown': sum(row['strict_unknown'] for row in rows), 'total': len(rows),
              'limitations': ['Synthetic names and questions; not an independent population sample.',
                              'Exact scoring may count alternative valid refusals as failures.',
                              'No evidence about names used in pretrained data or earlier project work.']}
    resume.write_json(OUT / 'unknown-audit.json', report)
    print(json.dumps({'strict_unknown': report['strict_unknown'], 'total': report['total']}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--unknown-audit']:
        audit_unknown_people()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--unknown-audit]')
