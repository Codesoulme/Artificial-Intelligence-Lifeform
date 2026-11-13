"""Test base-output preservation during online QA memory editing."""
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

OUT = ROOT / 'runs/deepseek-spora-online-qa-guard-20261114'
WEIGHT = 1.0


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior_path = baseline.OUT / 'result.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if prior['base_sha256'] != base_sha or base_sha != p.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.manual_seed(baseline.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
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
    guard_questions = baseline.GENERIC + (baseline.UNKNOWN,)
    guards = []
    for question in guard_questions:
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.no_grad():
            teacher = model(**prompt, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
            teacher_probs = teacher.softmax(-1).detach()
        guards.append({'question': question, 'prompt': prompt, 'probs': teacher_probs})
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in baseline.GENERIC]
    optimizer = torch.optim.AdamW(parameters, lr=baseline.LR, weight_decay=0, foreach=False)
    current, stages = {}, []
    for stage, (name, value) in enumerate(baseline.EVENTS):
        current[name] = value
        newest = baseline.make_training_example(tokenizer, name, value)
        others = [baseline.make_training_example(tokenizer, other, other_value)
                  for other, other_value in current.items() if other != name]
        trace = []
        for step in range(baseline.STEPS):
            use_replay = bool(others) and step % 4 == 3
            item = others[(step // 4) % len(others)] if use_replay else newest
            guard = guards[step % len(guards)]
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=item['inputs'], use_cache=False,
                           logits_to_keep=item['targets'].shape[1], return_dict=True)
            task_loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                        item['targets'].reshape(-1))
            (task_loss).backward()
            guard_logits = model(**guard['prompt'], use_cache=False, logits_to_keep=1,
                                 return_dict=True).logits[:, -1].float()
            kl = F.kl_div(guard_logits.log_softmax(-1), guard['probs'], reduction='batchmean')
            if not bool(torch.isfinite(task_loss)) or not bool(torch.isfinite(kl)):
                raise ValueError('Nonfinite objective')
            (WEIGHT * kl).backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step in (0, baseline.STEPS-1):
                trace.append({'step': step+1, 'task_loss': float(task_loss.detach()),
                              'guard_kl': float(kl.detach()), 'gradient_norm': float(grad)})
        answers = []
        for entity, expected in current.items():
            for kind, template in (('trained', baseline.TRAIN_TEMPLATE), ('held', baseline.HELD_TEMPLATE)):
                question = template.format(name=entity)
                answer = pilot.generate(model, tokenizer, question)
                answers.append({'name': entity, 'kind': kind, 'question': question, 'expected': expected,
                                'result': answer, 'exact': baseline.exact(answer['answer'], expected)})
        unknown = pilot.generate(model, tokenizer, baseline.UNKNOWN)
        generic = [pilot.generate(model, tokenizer, question) for question in baseline.GENERIC]
        stages.append({'stage': stage, 'event': {'name': name, 'value': value}, 'current': dict(current),
                       'trace': trace, 'answers': answers, 'unknown_answer': unknown, 'generic_answers': generic})
        print(json.dumps({'stage': stage, 'trace': trace,
                          'answers': [{'name': x['name'], 'kind': x['kind'], 'answer': x['result']['answer'],
                                       'exact': x['exact']} for x in answers],
                          'unknown': unknown['answer'], 'generic': [x['answer'] for x in generic]},
                         ensure_ascii=False), flush=True)
    for module in modules:
        module.enabled = False
    disabled = [pilot.generate(model, tokenizer, question) for question in baseline.GENERIC]
    if disabled != ordinary_before or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Disabled path or protected base changed')
    OUT.mkdir()
    torch.save({'format': 'online-qa-guard-pilot-v1', 'base_sha256': base_sha,
                'layers': online.LAYERS, 'rank': online.RANK,
                'adapters': [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                              if key.startswith(('down.', 'up.'))} for module in modules]}, OUT / 'adapter.pt')
    final = stages[-1]
    summary = {kind: {'cases': sum(x['kind'] == kind for x in final['answers']),
                      'exact': sum(x['exact'] for x in final['answers'] if x['kind'] == kind)}
               for kind in ('trained', 'held')}
    summary['ordinary_tokens_match_base'] = sum(a['token_ids'] == b['token_ids']
                                                for a, b in zip(final['generic_answers'], ordinary_before))
    report = {'status': 'online_qa_replay_guard_pilot', 'summary': summary, 'results': stages,
              'prior_replay_summary': prior['summary']['replay'], 'ordinary_before': ordinary_before,
              'guard_questions': guard_questions, 'guard_weight': WEIGHT, 'base_sha256': base_sha,
              'adapter_sha256': p.sha(OUT / 'adapter.pt'), 'prior_result_sha256': p.sha(prior_path),
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three preservation prompts are also evaluation prompts: not an independent test.',
                              'First-token KL does not enforce full-generation or semantic equivalence.',
                              'Unknown baseline is not guaranteed to be a correct refusal.',
                              'Facts, QA supervision, current values and replay selection are manually supplied.',
                              'Two entities, three events, one held wording and no durable individual.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
