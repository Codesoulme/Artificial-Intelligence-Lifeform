"""Compare correction and continuation from the same trained adapter state.

The experiment explicitly supplies fact labels and unknown examples. It does
not use SPORA's persistent individual or claim autonomous learning.
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta


OUT = ROOT / 'runs/deepseek-spora-incremental-correction-control-20261116.json'
SEED = 41907
STEPS = 72
TRAIN_TEMPLATES = ('请只回答{name}喜欢喝什么饮料。', '记录中{name}偏爱哪种饮品？')
HELD_TEMPLATES = ('{name}最喜欢的饮料是什么？', '只说出{name}的饮品偏好。')
TRAIN_UNKNOWN = ('小吴', '小孙', '小郑', '阿明')
HELD_UNKNOWN = ('小赵', '小周')
CONTROL_TRAIN = '你好，请简单介绍自己。'
ORDINARY = ('二加二等于多少？', '请写一句鼓励学习的话。')
FACT_STATES = {'before': {'小林': '咖啡', '阿宁': '红茶', '小雨': '果汁'},
               'continued_control': {'小林': '咖啡', '阿宁': '红茶', '小雨': '果汁'},
               'corrected': {'小林': '酸奶', '阿宁': '红茶', '小雨': '果汁'}}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in (18, 23):
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    initial = [{key: value.detach().clone() for key, value in module.state_dict().items()
                if key.startswith(('down.', 'up.'))} for module in modules]
    parameters = [parameter for module in modules for part in (module.down, module.up)
                  for parameter in part.parameters()]

    def prefix(question):
        return pilot.closed_think_inputs(tokenizer, question, 'cuda')['input_ids']

    def answer_loss(question, answer):
        inputs = prefix(question)
        ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor([ids], dtype=torch.long, device='cuda')
        full = torch.cat((inputs, target[:, :-1]), dim=1)
        scores = model(input_ids=full, use_cache=False, return_dict=True).logits
        scores = scores[:, inputs.shape[1] - 1:inputs.shape[1] - 1 + target.shape[1]].float()
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))

    def first_logits(inputs):
        return model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()

    control = prefix(CONTROL_TRAIN)
    with torch.no_grad():
        initial_control = first_logits(control).detach()
    generation_cases = [(group, template.format(name=name), name)
                        for name in FACT_STATES['before']
                        for group, template in (('known_train', TRAIN_TEMPLATES[0]),
                                                ('known_held', HELD_TEMPLATES[0]))]
    generation_cases += [('unknown_held', HELD_TEMPLATES[0].format(name=name), name)
                         for name in HELD_UNKNOWN]
    generation_cases += [('ordinary', question, None) for question in ORDINARY]
    with torch.inference_mode():
        off = {question: pilot.generate(model, tokenizer, question) for _, question, _ in generation_cases}

    results = []
    before_adapter = None
    before_optimizer = None
    for state_name, facts in FACT_STATES.items():
        starting = initial if state_name == 'before' else before_adapter
        for module, state in zip(modules, starting):
            module.down.load_state_dict({'weight': state['down.weight']}, strict=True)
            module.up.load_state_dict({'weight': state['up.weight']}, strict=True)
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0, foreach=False)
        if state_name != 'before':
            optimizer.load_state_dict(copy.deepcopy(before_optimizer))
        names = tuple(facts)
        for step in range(STEPS):
            name = names[(step // len(TRAIN_TEMPLATES)) % len(names)]
            template_index = step % len(TRAIN_TEMPLATES)
            question = TRAIN_TEMPLATES[template_index].format(name=name)
            unknown = TRAIN_UNKNOWN[(step // len(TRAIN_TEMPLATES)) % len(TRAIN_UNKNOWN)]
            unknown_question = TRAIN_TEMPLATES[template_index].format(name=unknown)
            optimizer.zero_grad(set_to_none=True)
            loss = answer_loss(question, facts[name])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite fact loss')
            loss.backward()
            (0.5 * answer_loss(unknown_question, '我不知道。')).backward()
            divergence = F.kl_div(F.log_softmax(first_logits(control), -1),
                                  F.softmax(initial_control, -1), reduction='batchmean')
            (0.5 * divergence).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 24 == 0:
                print(json.dumps({'state': state_name, 'step': step + 1,
                                  'fact_loss': float(loss.detach()),
                                  'control_kl': float(divergence.detach())}), flush=True)
        if state_name == 'before':
            before_adapter = [{key: value.detach().clone() for key, value in module.state_dict().items()
                               if key.startswith(('down.', 'up.'))} for module in modules]
            before_optimizer = copy.deepcopy(optimizer.state_dict())
        rows = []
        for group, question, name in generation_cases:
            answer = pilot.generate(model, tokenizer, question)
            expected = facts.get(name) if name is not None else None
            rows.append({'group': group, 'name': name, 'question': question,
                         'expected': expected, 'answer': answer, 'no_adapter': off[question],
                         'exact': expected is not None and answer['answer'].strip().rstrip('。.!！ ') == expected})
            print(json.dumps({'state': state_name, 'group': group, 'name': name,
                              'answer': answer['answer']}, ensure_ascii=True), flush=True)
        results.append({'state': state_name, 'facts': facts, 'rows': rows})
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base weights changed')
    pairs = []
    for left, right in zip(results[1]['rows'], results[2]['rows']):
        if left['question'] != right['question']:
            raise ValueError('Unmatched question')
        pairs.append({'group': left['group'], 'name': left['name'], 'question': left['question'],
                      'changed': left['answer']['token_ids'] != right['answer']['token_ids'],
                      'both_exact': bool(left['exact'] and right['exact'])})
    report = {'status': 'incremental_correction_control_development', 'states': results, 'pairs': pairs,
              'training_templates': TRAIN_TEMPLATES, 'held_templates': HELD_TEMPLATES,
              'training_unknown_names': TRAIN_UNKNOWN, 'held_unknown_names': HELD_UNKNOWN,
              'steps_per_state': STEPS, 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Explicit labeled facts and unknown subjects, not autonomous observation.',
                              'The correction branch replays the full current fact set for 72 steps.',
                              'The control continues the same pre-correction state and optimizer for 72 steps.',
                              'Two values, three known people and two held unknown people; development data.',
                              'Only one held template used in generation; no independent chat benchmark.',
                              'The adapter weights are not the original SPORA individual or base weights.',
                              'No persistent reload or historical-answer evaluation.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
