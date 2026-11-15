"""Audit full-answer edits with replay, KL, and one unknown-fact example."""
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
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta, observation_example


OUT = ROOT / 'runs/deepseek-spora-full-answer-retention-audit-20261116.json'
SEED = 41907
STEPS = 24
LR = 0.001
EVENTS = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小林', '咖啡'))
PARAPHRASES = ('请只回答{name}喜欢喝什么饮料。', '记录中{name}偏爱哪种饮品？')
CONTROLS = ('你好，请简单介绍自己。', '二加二等于多少？', '请写一句鼓励学习的话。')
GENERATION_CASES = (
    ('known_train_wording', '请只回答小林喜欢喝什么饮料。', '咖啡'),
    ('known_train_wording', '请只回答阿宁喜欢喝什么饮料。', '豆浆'),
    ('known_rephrase', '记录中小林偏爱哪种饮品？', '咖啡'),
    ('known_rephrase', '记录中阿宁偏爱哪种饮品？', '豆浆'),
    ('known_new_wording', '小林最喜欢的饮料是什么？', '咖啡'),
    ('known_new_wording', '阿宁最喜欢的饮料是什么？', '豆浆'),
    ('unknown', '请只回答小赵喜欢喝什么饮料。', None),
    ('unknown', '记录中小赵偏爱哪种饮品？', None),
    ('ordinary', '二加二等于多少？', None),
    ('ordinary', '请写一句鼓励学习的话。', None),
)
KL_WEIGHT = 0.5
REPLAY_WEIGHT = 0.5
UNKNOWN_WEIGHT = 0.5
UNKNOWN_QUESTION = '请只回答小吴喜欢喝什么饮料。'
UNKNOWN_ANSWER = '我不知道。'


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

    def logits(ids):
        return model(input_ids=ids, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()

    def answer_loss(prefix, answer):
        ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor([ids], dtype=torch.long, device='cuda')
        full = torch.cat((prefix, target[:, :-1]), dim=1)
        scores = model(input_ids=full, use_cache=False, return_dict=True).logits
        scores = scores[:, prefix.shape[1] - 1:prefix.shape[1] - 1 + target.shape[1]].float()
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))

    control_inputs = [pilot.closed_think_inputs(tokenizer, text, 'cuda')['input_ids'] for text in CONTROLS]
    with torch.no_grad():
        reference = [logits(ids).detach() for ids in control_inputs]
    unknown_prefix = pilot.closed_think_inputs(tokenizer, UNKNOWN_QUESTION, 'cuda')['input_ids']

    def evaluate(facts):
        rows = []
        with torch.inference_mode():
            for name, value in facts.items():
                target = tokenizer.encode(value, add_special_tokens=False)[0]
                for template in PARAPHRASES:
                    question = template.format(name=name)
                    scores = logits(pilot.closed_think_inputs(tokenizer, question, 'cuda')['input_ids'])[0]
                    rows.append({'name': name, 'value': value, 'question': question,
                                 'target_token_id': target,
                                 'rank': int((scores > scores[target]).sum()) + 1,
                                 'winner_token_id': int(scores.argmax())})
            divergences = []
            for ids, ref in zip(control_inputs, reference):
                current = logits(ids)
                divergences.append(float(F.kl_div(F.log_softmax(current, -1),
                                                  F.softmax(ref, -1), reduction='batchmean')))
        return {'facts': rows, 'control_kl_from_initial': divergences}

    arms = []
    for arm in ('replay_kl',):
        for module, state in zip(modules, initial):
            module.down.load_state_dict({'weight': state['down.weight']}, strict=True)
            module.up.load_state_dict({'weight': state['up.weight']}, strict=True)
        optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=0, foreach=False)
        facts = {}
        stages = []
        for name, value in EVENTS:
            fact, _, _ = observation_example(tokenizer, name, value)
            prefix = pilot.closed_think_inputs(tokenizer, PARAPHRASES[0].format(name=name), 'cuda')['input_ids']
            replay = [(old_name, old_value) for old_name, old_value in facts.items() if old_name != name]
            for step in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                loss = answer_loss(prefix, value)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite edit loss')
                loss.backward()
                if arm in ('replay', 'replay_kl') and replay:
                    old_name, old_value = replay[step % len(replay)]
                    old_prefix = pilot.closed_think_inputs(tokenizer, PARAPHRASES[0].format(name=old_name), 'cuda')['input_ids']
                    (REPLAY_WEIGHT * answer_loss(old_prefix, old_value)).backward()
                (UNKNOWN_WEIGHT * answer_loss(unknown_prefix, UNKNOWN_ANSWER)).backward()
                if arm in ('kl', 'replay_kl'):
                    current = logits(control_inputs[0])
                    divergence = F.kl_div(F.log_softmax(current, -1), F.softmax(reference[0], -1), reduction='batchmean')
                    (KL_WEIGHT * divergence).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            facts[name] = value
            stages.append({'observed_fact': fact, 'replay_facts': replay, 'training_loss': float(loss.detach()),
                           'evaluation': evaluate(facts)})
            print(json.dumps({'arm': arm, 'fact': fact, 'loss': float(loss.detach())}, ensure_ascii=True), flush=True)
        arms.append({'arm': arm, 'stages': stages})
    generation = []
    for group, question, expected in GENERATION_CASES:
        for module in modules:
            module.enabled = False
        off = pilot.generate(model, tokenizer, question)
        for module in modules:
            module.enabled = True
        on = pilot.generate(model, tokenizer, question)
        generation.append({'group': group, 'question': question, 'expected': expected,
                           'disabled': off, 'enabled': on,
                           'strict_exact': expected is not None and on['answer'].strip().rstrip('。.!！ ') == expected,
                           'changed': off['token_ids'] != on['token_ids']})
        print(json.dumps({'generation_group': group, 'question': question,
                          'enabled': on['answer']}, ensure_ascii=True), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base weights changed')
    report = {'status': 'controlled_full_answer_retention_free_generation', 'arms': arms,
              'generation': generation,
              'events': EVENTS, 'steps_per_event': STEPS, 'learning_rate': LR,
              'kl_weight': KL_WEIGHT, 'replay_weight': REPLAY_WEIGHT,
              'unknown_weight': UNKNOWN_WEIGHT, 'unknown_question': UNKNOWN_QUESTION,
              'unknown_answer': UNKNOWN_ANSWER,
              'base_sha256': base_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only two synthetic known entities and one unknown entity.',
                              'One question wording and one different unknown subject were used in training.',
                              'The rest are development probes, not a sealed benchmark.',
                              'The event value and its boundary are supplied by the experiment.',
                              'Disabled adapter is a base-path comparison, not an independent quality benchmark.',
                              'KL trains on one ordinary prompt; other prompts are held out.',
                              'No persistent individual or autonomous edit acceptance.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
