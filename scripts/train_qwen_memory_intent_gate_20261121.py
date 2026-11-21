"""Train and audit a Qwen-side gate for opt-in SPORA memory reads."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn

from audit_awake_reasoning_residual_holdout_20261121 import load_scenarios
from scripts.open_base_addressed_alignment import HELD_PROMPTS, TRAIN_PROMPTS, QUERY_PATH, QUERY_SHA, neural_addresses
from scripts.open_base_baseline import INDIVIDUAL, MODEL, INDIVIDUAL_SHA, WEIGHT_SHA, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS, variants
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


ORDINARY_TRAIN = (
    '你好，请简单介绍一下你能做什么。',
    '我今天有点累，给我一个放松建议。',
    '社会主义核心价值观的个人层面包括哪些内容？',
    '解释一下春天为什么会有花开。',
    '帮我写一句礼貌的感谢话。',
    '把这句话改得更简短：今天下午我要去图书馆。',
    '给我列出三种健康的早餐。',
    '为什么下雨后地面会变湿？',
    '写一段关于海边的短描述。',
    '请把“Good morning”翻译成中文。',
    '如何安排一天的学习时间？',
    '用一句话鼓励准备考试的朋友。',
)

ORDINARY_PROBE = (
    '面向对象编程里的对象和类有什么区别？',
    '请记录一下：明天上午十点开会。',
    '这个研究对象的定义是什么？',
    '帮我整理一份会议记录的格式。',
)

MEMORY_NATURAL_PROBE = (
    '我之前记录的对象{key}现在是多少？',
    '从记录里看，对象{key}当前数字是几？',
)

MEMORY_GATE_TEMPLATES = TRAIN_PROMPTS + MEMORY_NATURAL_PROBE + (
    '你记得编号{key}保存的数值吗？',
)


class IntentGate(nn.Module):
    format = 'spora-qwen-memory-intent-gate-v1'

    def __init__(self, width, cutoff=.5):
        super().__init__()
        self.head = nn.Linear(width, 1)
        self.cutoff = float(cutoff)

    def forward(self, features):
        return self.head(features).squeeze(-1)


@torch.inference_mode()
def hidden_features(model, tokenizer, questions):
    rows = []
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.model.layers[5].register_forward_hook(capture)
    for question in questions:
        try:
            rendered = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': question}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
            model(**inputs, use_cache=False)
            if len(captured) != 1:
                raise RuntimeError(f'Layer feature capture returned {len(captured)} outputs')
            hidden = captured.pop()[0].float()
            rows.append(torch.cat((hidden[-1], hidden.mean(dim=0)), dim=0) / 2**.5)
        except Exception:
            handle.remove()
            raise
    handle.remove()
    return torch.stack(rows).cuda()


def memory_questions(keys, prompts):
    return [template.format(key=key) for key in keys for template in prompts]


def make_runtime_gate(gate, cutoff):
    def decide(hidden):
        tokens = hidden[0].float()
        features = torch.cat((tokens[-1], tokens.mean(dim=0)), dim=0) / 2**.5
        score = gate(features.unsqueeze(0)).sigmoid().item()
        return score >= cutoff
    return decide


@torch.inference_mode()
def generate(model, tokenizer, question, *, adapter=None, state=None, addresses=None,
             layers=None, gate=None, limit=48):
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
    kwargs = dict(
        input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
        do_sample=False, max_new_tokens=limit, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if adapter is None:
        result = model.generate(**kwargs)
    else:
        with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses, gate=gate):
            result = model.generate(**kwargs)
    tail = result[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(tail, skip_special_tokens=True).strip(), bool(
        tail.numel() and tail[-1].item() == tokenizer.eos_token_id)


def train_gate(features, labels, steps, learning_rate):
    gate = IntentGate(features.shape[1]).cuda()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=0.)
    target = labels.to(device='cuda', dtype=torch.float32)
    losses = []
    for step in range(steps):
        logits = gate(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return gate.eval().requires_grad_(False), losses


@torch.inference_mode()
def gate_scores(gate, features):
    return gate(features).sigmoid().detach().cpu().tolist()


def main(output, steps, learning_rate, cutoff):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = (MODEL / 'model.safetensors', MODEL / 'config.json', MODEL / 'tokenizer.json',
               MODEL / 'tokenizer_config.json', INDIVIDUAL,
               ROOT / 'runs/open-base-addressed-full-answer-20261109/adapter.pt',
               ROOT / 'data/conversation-development-v2.json', QUERY_PATH)
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    torch.manual_seed(49107)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager',
    ).to('cuda').eval()
    model.requires_grad_(False)
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    query_model, _, _ = load(QUERY_PATH, 'cpu')
    memory_states, _ = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    adapter_payload = torch.load(
        ROOT / 'runs/open-base-addressed-full-answer-20261109/adapter.pt',
        map_location='cpu', weights_only=True)
    if adapter_payload.get('format') != 'spora-open-base-p26-full-answer-v1':
        raise ValueError('Unexpected memory adapter format')
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(adapter_payload['state'], strict=True)
    adapter.eval()
    layers = [model.model.layers[5], model.model.layers[17]]

    train_memory = memory_questions(tuple(key for pair in TRAIN_PAIRS for key in pair), MEMORY_GATE_TEMPLATES)
    ordinary_train = list(ORDINARY_TRAIN) * (len(train_memory) // len(ORDINARY_TRAIN))
    train_questions = ordinary_train + train_memory
    labels = torch.tensor([0] * len(ordinary_train) + [1] * len(train_memory))
    order = torch.randperm(len(train_questions), generator=torch.Generator().manual_seed(49107))
    train_questions = [train_questions[i] for i in order.tolist()]
    labels = labels[order]
    features = hidden_features(model, tokenizer, train_questions).clone()
    gate, losses = train_gate(features, labels, steps, learning_rate)
    runtime_gate = make_runtime_gate(gate, cutoff)

    train_scores = gate_scores(gate, features)
    train_pred = [score >= cutoff for score in train_scores]
    train_accuracy = sum(pred == bool(label) for pred, label in zip(train_pred, labels.tolist(), strict=True))

    scenarios = load_scenarios()
    ordinary_holdout = [question for scenario in scenarios for question in scenario['turns']]
    memory_holdout = memory_questions(tuple(key for pair in HELD_PAIRS for key in pair), HELD_PROMPTS)
    ordinary_scores = gate_scores(gate, hidden_features(model, tokenizer, ordinary_holdout))
    memory_scores = gate_scores(gate, hidden_features(model, tokenizer, memory_holdout))
    ordinary_probe_scores = gate_scores(gate, hidden_features(model, tokenizer, ORDINARY_PROBE))
    natural_memory_questions = memory_questions(
        tuple(key for pair in HELD_PAIRS for key in pair), MEMORY_NATURAL_PROBE)
    natural_memory_scores = gate_scores(gate, hidden_features(model, tokenizer, natural_memory_questions))
    ordinary_decisions = [score >= cutoff for score in ordinary_scores]
    memory_decisions = [score >= cutoff for score in memory_scores]

    ordinary_rows = []
    for question, score, use_memory in zip(ordinary_holdout, ordinary_scores, ordinary_decisions, strict=True):
        plain, plain_eos = generate(model, tokenizer, question)
        gated, gated_eos = generate(
            model, tokenizer, question,
            adapter=adapter, state=memory_states[HELD_PAIRS[0]][0], layers=layers,
            gate=runtime_gate,
        )
        ordinary_rows.append({
            'question': question, 'score': score, 'memory_enabled': use_memory,
            'plain': plain, 'gated': gated, 'plain_eos': plain_eos, 'gated_eos': gated_eos,
            'unchanged': plain == gated,
        })

    memory_rows = []
    for question, score, use_memory in zip(memory_holdout, memory_scores, memory_decisions, strict=True):
        pair = HELD_PAIRS[(len(memory_rows) // (len(HELD_PROMPTS) * 2))]
        state = memory_states[pair][0]
        addresses, parsed = neural_addresses(query_model, organism, question)
        answer, eos = generate(
            model, tokenizer, question,
            adapter=adapter, state=state, addresses=addresses, layers=layers,
            gate=runtime_gate,
        )
        memory_rows.append({
            'question': question, 'score': score, 'memory_enabled': use_memory,
            'parsed': parsed, 'answer': answer, 'eos': eos,
        })

    if (before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL / 'model.safetensors') != WEIGHT_SHA
            or sha(QUERY_PATH) != QUERY_SHA
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Protected input changed')
    torch.save({
        'format': IntentGate.format, 'cutoff': cutoff,
        'model_weight_sha256': WEIGHT_SHA, 'state': gate.state_dict(),
    }, output / 'intent_gate.pt')
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump({
            'format': 'spora-qwen-memory-intent-gate-diagnostic-v1',
            'steps': steps, 'learning_rate': learning_rate, 'cutoff': cutoff,
            'train_rows': len(train_questions), 'train_accuracy': train_accuracy,
            'ordinary_holdout_rows': len(ordinary_holdout),
            'ordinary_memory_enabled': sum(ordinary_decisions),
            'ordinary_unchanged': sum(row['unchanged'] for row in ordinary_rows),
            'memory_holdout_rows': len(memory_holdout),
            'memory_enabled': sum(memory_decisions),
            'ordinary_probe': [
                {'question': question, 'score': score, 'memory_enabled': score >= cutoff}
                for question, score in zip(ORDINARY_PROBE, ordinary_probe_scores, strict=True)
            ],
            'natural_memory_probe': [
                {'question': question, 'score': score, 'memory_enabled': score >= cutoff}
                for question, score in zip(natural_memory_questions, natural_memory_scores, strict=True)
            ],
            'losses': losses, 'ordinary_rows': ordinary_rows, 'memory_rows': memory_rows,
            'sources': {str(path): digest for path, digest in protected.items()},
            'seconds': time.perf_counter() - started,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'limitations': [
                'The gate is trained on a finite controlled memory-query vocabulary.',
                'The memory reader remains a finite numbered-task adapter.',
                'This candidate is not installed into the formal runtime or individual.',
            ],
        }, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'train_accuracy': train_accuracy, 'ordinary_memory_enabled': sum(ordinary_decisions),
           'ordinary_unchanged': sum(row['unchanged'] for row in ordinary_rows),
           'memory_enabled': sum(memory_decisions),
           'ordinary_probe_enabled': sum(score >= cutoff for score in ordinary_probe_scores),
           'natural_memory_probe_enabled': sum(score >= cutoff for score in natural_memory_scores)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--cutoff', type=float, default=.5)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve():
        parser.error('Output must be a new directory under runs')
    if args.steps < 1 or args.learning_rate <= 0 or not 0 < args.cutoff <= 1:
        parser.error('Invalid gate training arguments')
    main(output, args.steps, args.learning_rate, args.cutoff)
