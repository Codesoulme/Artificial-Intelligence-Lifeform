"""Audit additional correction updates on varied questions and ordinary prompts.

All facts and labels are synthetic and supplied by the experiment. This is a
development comparison, not an independent acceptance benchmark.
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
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_resume_equivalence_audit as resume
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-correction-generalization-20261116'
HELD_TEMPLATES = (data.HELD_TEMPLATES[0], data.HELD_TEMPLATES[1],
                  '如果只能选一种饮料，{name}会选择什么？',
                  '谈到饮品，{name}喜欢喝哪一种？')
STAGES = (0, 24, 60)


def score(answer, expected):
    return answer.strip().rstrip('。.!！ ') == expected


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    original = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    resume_meta = json.loads((resume.OUT / 'resume.json').read_text(encoding='utf-8'))
    source = resume.OUT / 'resume.pt'
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if (base_sha != base.WEIGHT_SHA or base_sha != original['base_sha256']
            or feature_sha != original['feature_sha256'] or files != original['files']
            or base.sha(source) != resume_meta['candidate_sha256']):
        raise ValueError('Source candidate or model binding mismatch')
    previous_training = json.loads((binding_source.OUT / 'training.json').read_text(encoding='utf-8'))
    prior_rows = {row['question']: row for row in previous_training['states'][-1]['rows']}
    prior_answers = {row['question']: row['answer'] for row in resume_meta['answers']}
    controls_train, controls_held = data.load_reviewed_prompts()
    controls = [('ordinary_train', row['prompt']) for row in controls_train]
    controls += [('ordinary_held', row['prompt']) for row in controls_held]
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
    params = [parameter for module in modules for part in (module.down, module.up)
              for parameter in part.parameters()]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=0, foreach=False)
    info = checkpoint.restore(source, modules, base_sha256=base_sha, feature_sha256=feature_sha,
                              layers=binding_source.LAYERS, optimizer=optimizer)
    if info['steps'] != 156:
        raise ValueError('Unexpected source step')

    def loss_for(question, answer):
        inputs = pilot.closed_think_inputs(tokenizer, question, 'cuda')['input_ids']
        tokens = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor([tokens], dtype=torch.long, device='cuda')
        full = torch.cat((inputs, target[:, :-1]), dim=1)
        scores = model(input_ids=full, use_cache=False, return_dict=True).logits
        scores = scores[:, inputs.shape[1] - 1:inputs.shape[1] - 1 + target.shape[1]].float()
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))

    cases = [(f'known_train_{index}', template.format(name=name), answer)
             for name, answer in resume.FACTS.items()
             for index, template in enumerate(data.TRAIN_TEMPLATES)]
    cases += [(f'known_held_{index}', template.format(name=name), answer)
              for name, answer in resume.FACTS.items()
              for index, template in enumerate(HELD_TEMPLATES)]
    cases += [(f'unknown_held_{index}', template.format(name=name), '我不知道')
              for name in data.HELD_UNKNOWN for index, template in enumerate(HELD_TEMPLATES)]
    cases += [(group, question, None) for group, question in controls]
    losses, stages = [], []
    names = tuple(resume.FACTS)
    for step in range(STAGES[-1] + 1):
        if step in STAGES:
            rows = []
            for group, question, expected in cases:
                answer = pilot.generate(model, tokenizer, question)
                row = {'group': group, 'question': question, 'expected': expected, 'answer': answer}
                if expected is not None:
                    row['exact'] = score(answer['answer'], expected)
                if question in prior_answers:
                    row['same_start'] = answer['token_ids'] == prior_answers[question]['token_ids']
                if question in prior_rows:
                    row['same_base'] = answer['token_ids'] == prior_rows[question]['no_adapter']['token_ids']
                rows.append(row)
            path = OUT / f'stage-{step}.pt'
            checkpoint.save(path, modules, optimizer, base_sha256=base_sha, feature_sha256=feature_sha,
                            layers=binding_source.LAYERS, steps=info['steps'] + step)
            stages.append({'additional_steps': step, 'candidate_sha256': base.sha(path), 'rows': rows})
            print(json.dumps({'stage': step, 'rows': len(rows)}, ensure_ascii=False), flush=True)
        if step == STAGES[-1]:
            break
        absolute = resume.TOTAL + step
        name = names[(absolute // len(data.TRAIN_TEMPLATES)) % len(names)]
        template = data.TRAIN_TEMPLATES[absolute % len(data.TRAIN_TEMPLATES)]
        unknown = data.TRAIN_UNKNOWN[(absolute // len(data.TRAIN_TEMPLATES)) % len(data.TRAIN_UNKNOWN)]
        optimizer.zero_grad(set_to_none=True)
        fact_loss = loss_for(template.format(name=name), resume.FACTS[name])
        fact_loss.backward()
        unknown_loss = loss_for(template.format(name=unknown), '我不知道。')
        (0.5 * unknown_loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({'step': step + 1, 'fact': float(fact_loss.detach()),
                       'unknown': float(unknown_loss.detach())})
        if (step + 1) % 12 == 0:
            print(json.dumps({'step': step + 1, 'fact_loss': losses[-1]['fact']}), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha or base.sha(source) != resume_meta['candidate_sha256']:
        raise ValueError('Base or source candidate changed')
    result = {'source_sha256': resume_meta['candidate_sha256'], 'base_sha256': base_sha,
              'held_templates': HELD_TEMPLATES, 'stages': stages, 'losses': losses,
              'seconds': time.perf_counter() - start_time,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic labels and a small repeated development set.',
                              'No independent benchmark, autonomous observation or historical retrieval.',
                              'No ordinary-prompt KL training during these updates.',
                              'Greedy generation capped at 24 tokens; strict exact short-answer scores.']}
    resume.write_json(OUT / 'result.json', result)
    print(json.dumps({'seconds': result['seconds']}), flush=True)


def verify():
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    original = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    stage = result['stages'][-1]
    candidate = OUT / f"stage-{stage['additional_steps']}.pt"
    feature_sha, files = binding_source.binding()
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    if (base_sha != base.WEIGHT_SHA or base_sha != result['base_sha256']
            or feature_sha != original['feature_sha256'] or files != original['files']
            or base.sha(candidate) != stage['candidate_sha256']):
        raise ValueError('Checkpoint or model binding mismatch')
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
    for original_row in stage['rows']:
        answer = pilot.generate(model, tokenizer, original_row['question'])
        rows.append({'group': original_row['group'], 'question': original_row['question'],
                     'answer': answer, 'same_tokens': answer['token_ids'] == original_row['answer']['token_ids']})
    if base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base changed during verification')
    report = {'candidate_sha256': stage['candidate_sha256'], 'rows': rows,
              'same_tokens': sum(row['same_tokens'] for row in rows), 'total': len(rows)}
    resume.write_json(OUT / 'verification.json', report)
    print(json.dumps({'same_tokens': report['same_tokens'], 'total': report['total']}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        verify()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--verify]')
