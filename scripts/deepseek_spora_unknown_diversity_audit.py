"""Matched replay comparison for unknown-person training diversity.

This experiment uses supplied labels and does not implement an autonomous
memory writer, identity verification, or a deployed individual.
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
from scripts import deepseek_spora_correction_generalization_audit as scoring
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_multirelation_correction_audit as prior
from scripts import deepseek_spora_resume_equivalence_audit as io
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-unknown-diversity-20261116'
STEPS = 84
EXTRA_TRAIN = ('阿川', '阿旭', '阿辰', '阿海', '阿志', '阿琪', '阿雪')
FRESH_HELD = ('小柏', '阿沐', '小舟', '阿朵')
EXTRA_PAIRS = tuple((name, relation) for name in EXTRA_TRAIN for relation in prior.TEMPLATES)


def unknown_pair(step, diverse):
    cycle = step // 14
    fact_index = (step // 2) % len(prior.CORRECTED)
    if diverse and cycle % 2:
        return EXTRA_PAIRS[((cycle // 2) * 7 + fact_index) % len(EXTRA_PAIRS)]
    return prior.UNKNOWN_TRAIN[(step // 2 + cycle) % len(prior.UNKNOWN_TRAIN)]


def validate_design():
    prior.validate_design()
    known = {name for name, _ in prior.CORRECTED}
    training = set(EXTRA_TRAIN) | {name for name, _ in prior.UNKNOWN_TRAIN}
    held = set(prior.AUDIT_NAMES) | set(FRESH_HELD)
    if (known & set(EXTRA_TRAIN) or known & held or training & held
            or set(prior.AUDIT_NAMES) & set(FRESH_HELD)):
        raise ValueError('Training, held and known names must be disjoint')
    if set(prior.UNKNOWN_TRAIN) & set(prior.CORRECTED):
        raise ValueError('Unknown relation conflicts with a known fact')


def main():
    validate_design()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous = json.loads((prior.OUT / 'result.json').read_text(encoding='utf-8'))
    binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    source = prior.OUT / 'corrected.pt'
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    source_sha = base.sha(source)
    if (base_sha != base.WEIGHT_SHA or base_sha != previous['base_sha256']
            or feature_sha != binding['feature_sha256'] or files != binding['files']
            or source_sha != previous['phases'][-1]['candidate_sha256']):
        raise ValueError('Base or candidate binding mismatch')
    train_controls, held_controls = data.load_reviewed_prompts()
    controls = [item['prompt'] for item in train_controls]
    ordinary = [('ordinary_train', item['prompt']) for item in train_controls]
    ordinary += [('ordinary_held', item['prompt']) for item in held_controls]
    before = {row['question']: row for row in previous['phases'][-1]['rows']}
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
    initial = checkpoint.restore(source, modules, base_sha256=base_sha, feature_sha256=feature_sha,
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

    control_tokens = [prefix(question) for question in controls]
    with torch.no_grad():
        teacher = [F.softmax(first_logits(item), -1).detach() for item in control_tokens]
    cases = [case for case in prior.examples(prior.CORRECTED)
             if case['group'] in ('old_train', 'old_held', 'new_train', 'new_held', 'unknown_relation')]
    for group, names in (('unknown_previous_audit', prior.AUDIT_NAMES), ('unknown_fresh', FRESH_HELD)):
        for name in names:
            for relation, templates in prior.TEMPLATES.items():
                for template in templates['held'][:2 if relation == 'drink' else 1]:
                    cases.append({'group': group, 'name': name, 'relation': relation,
                                  'question': template.format(name=name), 'expected': '我不知道'})
    cases += [{'group': group, 'question': question} for group, question in ordinary]
    outputs, losses = [], []
    fact_items = tuple(prior.CORRECTED.items())
    for branch, diverse in (('original', False), ('diverse', True)):
        checkpoint.restore(source, modules, base_sha256=base_sha, feature_sha256=feature_sha,
                           layers=binding_source.LAYERS, optimizer=optimizer)
        for step in range(STEPS):
            (name, relation), value = fact_items[(step // 2) % len(fact_items)]
            template_index = step % 2
            question = prior.TEMPLATES[relation]['train'][template_index].format(name=name)
            unknown_name, unknown_relation = unknown_pair(step, diverse)
            unknown_question = prior.TEMPLATES[unknown_relation]['train'][template_index].format(name=unknown_name)
            optimizer.zero_grad(set_to_none=True)
            fact_loss = answer_loss(question, value)
            fact_loss.backward()
            unknown_loss = answer_loss(unknown_question, '我不知道。')
            (0.5 * unknown_loss).backward()
            index = (step // 2) % len(control_tokens)
            kl = F.kl_div(F.log_softmax(first_logits(control_tokens[index]), -1), teacher[index],
                          reduction='batchmean')
            (0.5 * kl).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append({'branch': branch, 'step': step + 1, 'fact': float(fact_loss.detach()),
                           'unknown': float(unknown_loss.detach()), 'kl': float(kl.detach()),
                           'unknown_name': unknown_name, 'unknown_relation': unknown_relation})
            if (step + 1) % 28 == 0:
                print(json.dumps({'branch': branch, 'step': step + 1, 'fact': losses[-1]['fact']}), flush=True)
        rows = []
        for case in cases:
            answer = pilot.generate(model, tokenizer, case['question'])
            row = {**case, 'answer': answer}
            if 'expected' in case:
                row['exact'] = scoring.score(answer['answer'], case['expected'])
            if case['question'] in before:
                row['same_start'] = answer['token_ids'] == before[case['question']]['answer']['token_ids']
            rows.append(row)
        target = OUT / f'{branch}.pt'
        checkpoint.save(target, modules, optimizer, base_sha256=base_sha, feature_sha256=feature_sha,
                        layers=binding_source.LAYERS, steps=initial['steps'] + STEPS)
        outputs.append({'branch': branch, 'rows': rows, 'candidate_sha256': base.sha(target)})
        print(json.dumps({'branch': branch, 'evaluated': len(rows)}), flush=True)
    if base.sha(source) != source_sha or base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base or source candidate changed')
    report = {'source_sha256': source_sha, 'base_sha256': base_sha, 'steps_per_branch': STEPS,
              'extra_train_names': EXTRA_TRAIN, 'fresh_held_names': FRESH_HELD,
              'branches': outputs, 'losses': losses, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Controlled labels and full fact replay, not autonomous learning.',
                              'One source, one seed, small synthetic development set.',
                              'Previously audited names and prompts may influence experimental design.',
                              'Fresh names are not proven absent from prior model pretraining.',
                              'First-token KL is not a complete chat preservation constraint.',
                              'Strict short-answer matching and 24-token generation limit.']}
    io.write_json(OUT / 'result.json', report)
    print(json.dumps({'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
