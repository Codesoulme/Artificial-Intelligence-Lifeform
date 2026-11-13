"""Study negative identity supervision in isolated online QA parameters.

The training examples are explicitly constructed from an oracle list of
unknown identities. This is not autonomous enrollment or a deployed guard.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_online_lora_memory_pilot as online
from scripts import deepseek_spora_online_qa_replay_pilot as baseline
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-online-unknown-training-20261114'
TRAIN_UNKNOWN = ('小华', '阿涛', '小美', '小志')
HELD_UNKNOWN = ('小王', '小雪', '小赵')
STEPS = 64
UNKNOWN_VALUE = '不知道'
GUARD_WEIGHT = 1.0


def refusal(answer):
    text = answer.strip().rstrip('。.!！ ').strip()
    return text == UNKNOWN_VALUE


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if set(TRAIN_UNKNOWN) & set(HELD_UNKNOWN) or set(TRAIN_UNKNOWN) & {name for name, _ in baseline.EVENTS}:
        raise ValueError('Identity splits overlap')
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.manual_seed(baseline.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in online.LAYERS:
        mlp = model.model.layers[layer].mlp
        module = online.OnlineDelta(mlp.down_proj, online.RANK)
        mlp.down_proj = module
        modules.append(module)
    parameters = [param for module in modules for projection in (module.down, module.up)
                  for param in projection.parameters()]
    guards = []
    for question in baseline.GENERIC:
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.no_grad():
            logits = model(**prompt, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
        guards.append({'question': question, 'prompt': prompt, 'probs': logits.softmax(-1).detach()})
    ordinary_before = [pilot.generate(model, tokenizer, q) for q in baseline.GENERIC]
    unknown_examples = [baseline.make_training_example(tokenizer, name, UNKNOWN_VALUE) for name in TRAIN_UNKNOWN]
    optimizer = torch.optim.AdamW(parameters, lr=baseline.LR, weight_decay=0, foreach=False)
    current, stages = {}, []
    for stage, (name, value) in enumerate(baseline.EVENTS):
        current[name] = value
        newest = baseline.make_training_example(tokenizer, name, value)
        others = [baseline.make_training_example(tokenizer, other, other_value)
                  for other, other_value in current.items() if other != name]
        counts = {'new': 0, 'replay': 0, 'unknown': 0}
        trace = []
        for step in range(STEPS):
            if step % 4 == 3:
                item = unknown_examples[(step // 4 + stage) % len(unknown_examples)]
                counts['unknown'] += 1
            elif step % 4 == 2 and others:
                item = others[(step // 4) % len(others)]
                counts['replay'] += 1
            else:
                item = newest
                counts['new'] += 1
            guard = guards[step % len(guards)]
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=item['inputs'], use_cache=False,
                           logits_to_keep=item['targets'].shape[1], return_dict=True)
            task = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                   item['targets'].reshape(-1))
            if not bool(torch.isfinite(task)):
                raise ValueError('Nonfinite task loss')
            task.backward()
            guard_logits = model(**guard['prompt'], use_cache=False, logits_to_keep=1,
                                 return_dict=True).logits[:, -1].float()
            kl = F.kl_div(guard_logits.log_softmax(-1), guard['probs'], reduction='batchmean')
            if not bool(torch.isfinite(kl)):
                raise ValueError('Nonfinite preservation loss')
            (GUARD_WEIGHT * kl).backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step in (0, STEPS-1):
                trace.append({'step': step+1, 'task_loss': float(task.detach()),
                              'guard_kl': float(kl.detach()), 'gradient_norm': float(gradient)})
        answers = []
        for entity, expected in current.items():
            for kind, template in (('trained', baseline.TRAIN_TEMPLATE), ('held', baseline.HELD_TEMPLATE)):
                question = template.format(name=entity)
                answer = pilot.generate(model, tokenizer, question)
                answers.append({'name': entity, 'kind': kind, 'question': question, 'expected': expected,
                                'result': answer, 'exact': baseline.exact(answer['answer'], expected)})
        unknowns = []
        for entity in HELD_UNKNOWN:
            for kind, template in (('trained', baseline.TRAIN_TEMPLATE), ('held', baseline.HELD_TEMPLATE)):
                question = template.format(name=entity)
                answer = pilot.generate(model, tokenizer, question)
                unknowns.append({'name': entity, 'kind': kind, 'question': question, 'result': answer,
                                 'explicit_unknown': refusal(answer['answer']),
                                 'contains_known_value': any(v in answer['answer'] for v in current.values())})
        generic = [pilot.generate(model, tokenizer, q) for q in baseline.GENERIC]
        stages.append({'stage': stage, 'event': {'name': name, 'value': value}, 'current': dict(current),
                       'counts': counts, 'trace': trace, 'answers': answers, 'unknowns': unknowns,
                       'generic_answers': generic})
        print(json.dumps({'stage': stage, 'counts': counts,
                          'known_correct': sum(row['exact'] for row in answers),
                          'unknown_explicit': sum(row['explicit_unknown'] for row in unknowns),
                          'unknown_known_value': sum(row['contains_known_value'] for row in unknowns)},
                         ensure_ascii=False), flush=True)
    for module in modules:
        module.enabled = False
    if [pilot.generate(model, tokenizer, q) for q in baseline.GENERIC] != ordinary_before:
        raise ValueError('Disabled path changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Protected base changed')
    OUT.mkdir()
    torch.save({'format': 'online-unknown-training-v1', 'base_sha256': base_sha,
                'layers': online.LAYERS, 'rank': online.RANK,
                'adapters': [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                              if key.startswith(('down.', 'up.'))} for module in modules]}, OUT / 'adapter.pt')
    final = stages[-1]
    summary = {kind: {'cases': sum(row['kind'] == kind for row in final['answers']),
                      'exact': sum(row['exact'] for row in final['answers'] if row['kind'] == kind)}
               for kind in ('trained', 'held')}
    summary['unknown'] = {kind: {'cases': sum(row['kind'] == kind for row in final['unknowns']),
                                 'explicit_unknown': sum(row['explicit_unknown'] for row in final['unknowns'] if row['kind'] == kind),
                                 'contains_known_value': sum(row['contains_known_value'] for row in final['unknowns'] if row['kind'] == kind)}
                          for kind in ('trained', 'held')}
    summary['generic_token_matches_base'] = sum(a['token_ids'] == b['token_ids']
                                                for a, b in zip(final['generic_answers'], ordinary_before))
    report = {'status': 'online_unknown_supervision_pilot', 'summary': summary, 'stages': stages,
              'train_unknown_names': TRAIN_UNKNOWN, 'held_unknown_names': HELD_UNKNOWN,
              'steps_per_event': STEPS, 'guard_weight': GUARD_WEIGHT, 'ordinary_before': ordinary_before,
              'base_sha256': base_sha, 'adapter_sha256': p.sha(OUT / 'adapter.pt'),
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Unknown identities and labels are provided by the experiment, not model discovery.',
                              'Fixed synthetic questions and two known people; held names were used in earlier development.',
                              'A refusal may be semantically valid but not match the exact metric.',
                              'Higher total training steps and negative examples prevent a matched-compute baseline claim.',
                              'Unrecorded relations, aliases, shared names, natural writing and persistence are not tested.',
                              'Two generic preservation questions were also used in training.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
