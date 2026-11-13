"""Matched online QA alignment with and without current-fact replay."""
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
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-online-qa-replay-20261114'
EVENTS = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小林', '咖啡'))
TRAIN_TEMPLATE = '请只回答{name}喜欢喝什么饮料。'
HELD_TEMPLATE = '记录里{name}偏爱的饮品是什么？只说名称。'
GENERIC = ('请写一句鼓励学习的话。', '一加一等于几？')
UNKNOWN = '请只回答小赵喜欢喝什么饮料。'
STEPS = 48
LR = 0.001
SEED = 62419


def make_training_example(tokenizer, name, value):
    question = TRAIN_TEMPLATE.format(name=name)
    prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
    targets = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not targets[:-1]:
        raise ValueError('Empty value encoding')
    prefix = prompt['input_ids']
    continuation = torch.tensor([targets[:-1]], device='cuda', dtype=torch.long)
    inputs = torch.cat((prefix, continuation), dim=1)
    return {'name': name, 'value': value, 'inputs': inputs,
            'targets': torch.tensor([targets], device='cuda', dtype=torch.long)}


def exact(answer, value):
    return answer.strip().rstrip('。.!！ ').strip() == value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.manual_seed(SEED)
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
    initial = [{key: value.detach().clone() for key, value in module.state_dict().items()
                if key.startswith(('down.', 'up.'))} for module in modules]
    parameters = [param for module in modules for projection in (module.down, module.up)
                  for param in projection.parameters()]
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in GENERIC]
    results = {}
    payloads = {}
    for arm in ('new_only', 'replay'):
        for module, state in zip(modules, initial):
            module.load_state_dict(state, strict=False)
        optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=0, foreach=False)
        current = {}
        stages = []
        for stage, (name, value) in enumerate(EVENTS):
            current[name] = value
            newest = make_training_example(tokenizer, name, value)
            others = [make_training_example(tokenizer, other, other_value)
                      for other, other_value in current.items() if other != name]
            trace = []
            counts = {'new': 0, 'replay': 0}
            for step in range(STEPS):
                replay = arm == 'replay' and bool(others) and step % 4 == 3
                item = others[(step // 4) % len(others)] if replay else newest
                counts['replay' if replay else 'new'] += 1
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=item['inputs'], use_cache=False,
                               logits_to_keep=item['targets'].shape[1], return_dict=True)
                loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                       item['targets'].reshape(-1))
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                if step in (0, STEPS-1):
                    trace.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(grad)})
            answers = []
            for entity, expected in current.items():
                for kind, template in (('trained', TRAIN_TEMPLATE), ('held', HELD_TEMPLATE)):
                    question = template.format(name=entity)
                    answer = pilot.generate(model, tokenizer, question)
                    answers.append({'name': entity, 'kind': kind, 'question': question,
                                    'expected': expected, 'result': answer,
                                    'exact': exact(answer['answer'], expected)})
            unknown = pilot.generate(model, tokenizer, UNKNOWN)
            generic = [pilot.generate(model, tokenizer, question) for question in GENERIC]
            stages.append({'stage': stage, 'event': {'name': name, 'value': value},
                           'current': dict(current), 'counts': counts, 'trace': trace,
                           'answers': answers, 'unknown_answer': unknown, 'generic_answers': generic})
            print(json.dumps({'arm': arm, 'stage': stage, 'counts': counts,
                              'answers': [{'name': x['name'], 'kind': x['kind'],
                                           'answer': x['result']['answer'], 'exact': x['exact']}
                                          for x in answers]}, ensure_ascii=False), flush=True)
        results[arm] = stages
        payloads[arm] = [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                          if key.startswith(('down.', 'up.'))} for module in modules]
    for module in modules:
        module.enabled = False
    disabled = [pilot.generate(model, tokenizer, question) for question in GENERIC]
    if disabled != ordinary_before or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Disabled path or protected base changed')
    OUT.mkdir()
    torch.save({'format': 'online-qa-replay-pilot-v1', 'base_sha256': base_sha,
                'layers': online.LAYERS, 'rank': online.RANK, 'arms': payloads}, OUT / 'adapters.pt')
    summary = {}
    for arm, stages in results.items():
        final = stages[-1]
        summary[arm] = {kind: {'cases': sum(x['kind'] == kind for x in final['answers']),
                               'exact': sum(x['exact'] for x in final['answers'] if x['kind'] == kind)}
                        for kind in ('trained', 'held')}
        summary[arm]['ordinary_tokens_match_base'] = sum(
            a['token_ids'] == b['token_ids'] for a, b in zip(final['generic_answers'], ordinary_before))
    report = {'status': 'online_qa_replay_development_pilot', 'summary': summary,
              'events': EVENTS, 'results': results, 'ordinary_before': ordinary_before,
              'unknown_question': UNKNOWN, 'disabled_matches_base': True,
              'base_sha256': base_sha, 'adapter_sha256': p.sha(OUT / 'adapters.pt'),
              'steps_per_event': STEPS, 'lr': LR, 'seed': SEED,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['QA examples and replay identities/values are supplied by the experiment.',
                              'Replay excludes superseded values and does not verify historical recall.',
                              'Two entities, three events, two ordinary requests, one unknown and one held wording.',
                              'Equal optimizer steps do not imply identical compute or equivalent training examples.',
                              'Generated replies and low-rank parameters are not integrated with a durable SPORA individual.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
