"""Connect the bounded read planner to native decoder generation.

The planner chooses fields, order, grouping and response style.  The decoder
still generates the response; a request-local logits processor only supplies a
soft next-token guide derived from the chosen plan and live state.  No durable
individual or language checkpoint is modified.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer, LogitsProcessor

from audit_spora_dialogue_working_state import (
    FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator, state_from,
    verify_date_isolation,
)
from audit_spora_read_plan import DATA, LABELS, ReadPlan, decode_plan, render
from audit_spora_writer_transfer import write_metrics
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import _chat_inputs
from scripts.train_spora_dialogue_working_state import WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader


CONVERSATION = ROOT / 'data/conversation-development-v2.json'
BASELINE = ROOT / 'runs/spora-chat-backbones-v1/deepseek.json'
PLAN_FORMAT = 'spora-read-plan-v1'
WRITER_FORMAT = 'spora-working-turn-experiment-v1'


class PlanScores(LogitsProcessor):
    """Softly guide each generated position toward the rendered plan."""

    def __init__(self, prefix_length, target, weight):
        if not target:
            raise ValueError('A nonempty target sequence is required')
        if type(prefix_length) is not int or prefix_length < 1:
            raise ValueError('Invalid prompt length')
        if type(weight) not in (int, float) or not 0. <= float(weight) <= 64.:
            raise ValueError('Weight must be finite and in [0, 64]')
        self.prefix_length = prefix_length
        self.target = tuple(int(token) for token in target)
        self.weight = float(weight)

    def __call__(self, input_ids, scores):
        step = int(input_ids.shape[1]) - self.prefix_length
        if step < 0:
            raise ValueError('Generation prompt became shorter than its initial length')
        if step >= len(self.target):
            return scores
        guided = scores.clone()
        guided[:, self.target[step]] += self.weight
        return guided


def plan_text(tokenizer, kind, fields, plan):
    text = render(kind, fields, plan)
    return None if text is None else text


def target_tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]


@torch.inference_mode()
def native_generate(decoder, tokenizer, question, text, weight):
    inputs, _ = _chat_inputs(tokenizer, (), question, 384)
    inputs = inputs.to(decoder.base.device)
    length = int(inputs.input_ids.shape[1])
    processors = []
    if text is not None and weight > 0.:
        processors.append(PlanScores(length, target_tokens(tokenizer, text), weight))
    output = decoder.generate(
        **inputs, use_cache=True, do_sample=False, max_new_tokens=40,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        logits_processor=processors,
    )[0, length:]
    tokens = output.tolist()
    return {
        'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        'ended': bool(tokens and tokens[-1] == tokenizer.eos_token_id),
        'tokens': len(tokens),
    }


def output_metrics(answer, ended, kind, fields, plan, expected):
    if plan is None:
        return {
            'renderable': False, 'fields': False, 'order': False,
            'group': False, 'style': False, 'exact': False,
        }
    order, joined, style = plan
    if any(field not in fields for field in order):
        return {
            'renderable': False, 'fields': False, 'order': False,
            'group': False, 'style': False, 'exact': False,
        }
    values = [fields[field] for field in order]
    rendered_values = len(values) - 1 if joined else len(values)
    positions = []
    cursor = 0
    for value in values:
        position = answer.find(value, cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(value)
    fields_ok = all(position >= 0 for position in positions)
    order_ok = fields_ok and all(left <= right for left, right in zip(positions, positions[1:]))
    if all(field in fields for field in ('day', 'time')):
        group_value = ''.join(fields[field] for field in ('day', 'time'))
        group_ok = (group_value in answer) if joined else (group_value not in answer)
    else:
        group_ok = not joined
    labels = [LABELS[kind][field] for field in order]
    if style == 'only':
        style_ok = '\n' not in answer and ' / ' not in answer and '，' not in answer
    elif style == 'slash':
        style_ok = (answer.count(' / ') == rendered_values - 1 and '\n' not in answer
                    and '，' not in answer)
    elif style == 'comma':
        style_ok = (answer.count('，') == rendered_values - 1 and '\n' not in answer
                    and ' / ' not in answer)
    else:
        lines = answer.splitlines()
        style_ok = len(lines) == len(order) and all(
            line.startswith(label + '：') for line, label in zip(lines, labels, strict=True))
    return {
        'renderable': True, 'fields': fields_ok, 'order': order_ok,
        'group': group_ok, 'style': style_ok,
        'exact': bool(ended and answer == expected),
    }


def summarize(rows):
    keys = ('renderable', 'fields', 'order', 'group', 'style', 'ended', 'exact')
    return {key: sum(bool(row.get(key, False)) for row in rows) for key in keys}


def ordinary_reply(decoder, tokenizer, history, question):
    messages = [*history, {'role': 'user', 'content': question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith('<think>\n'):
        raise ValueError('Unexpected DeepSeek chat template ending')
    inputs = tokenizer(prompt + '</think>\n\n', add_special_tokens=False, return_tensors='pt')
    inputs = inputs.to(decoder.base.device)
    length = int(inputs.input_ids.shape[1])
    tokens = decoder.generate(
        **inputs, use_cache=True, do_sample=False, max_new_tokens=64,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )[0, length:].tolist()
    return {
        'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id),
        'tokens': len(tokens),
    }


def ordinary_audit(decoder, tokenizer):
    development = json.loads(CONVERSATION.read_text(encoding='utf-8'))['scenarios']
    prior = json.loads(BASELINE.read_text(encoding='utf-8'))['rollouts']
    rows = []
    for scenario in development:
        history = []
        for index, question in enumerate(scenario['turns']):
            current = ordinary_reply(decoder, tokenizer, history, question)
            reference = prior[scenario['name']][index]
            rows.append({
                'scenario': scenario['name'], 'turn': index, 'question': question,
                'current': current, 'reference': {
                    'answer': reference['answer'], 'eos': reference['eos'],
                }, 'answer_same': current['answer'] == reference['answer'],
                'eos_same': current['eos'] == reference['eos'],
            })
            history.extend([{'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': current['answer']}])
    return {
        'summary': {
            'turns': len(rows),
            'current_eos': sum(row['current']['eos'] for row in rows),
            'reference_eos': sum(row['reference']['eos'] for row in rows),
            'answer_same': sum(row['answer_same'] for row in rows),
            'eos_same': sum(row['eos_same'] for row in rows),
        },
        'rows': rows,
    }


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (
        DATA, CONVERSATION, BASELINE, args.checkpoint, args.database, args.writer, args.planner,
    )}
    held = json.loads(DATA.read_text(encoding='utf-8'))
    writer_payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    planner_payload = torch.load(args.planner, map_location='cpu', weights_only=True)
    if (writer_payload['format'], writer_payload['parent_sha256'], writer_payload['fields'],
        writer_payload['feature_depth'], writer_payload['writer_width']) != (
            WRITER_FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    if (planner_payload['format'], planner_payload['parent_sha256']) != (
            PLAN_FORMAT, sources[args.checkpoint]):
        raise ValueError('Planner checkpoint is incompatible with the parent')

    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=writer_payload['hidden']).to(device)
        writer.load_state_dict(writer_payload['writer'])
        locator = SemanticLocator(writer.eval(), features)
        hidden = planner_payload['model']['project.weight'].shape[1]
        planner = ReadPlan(hidden).to(device)
        planner.load_state_dict(planner_payload['model'])
        planner.eval()

        questions = [row['question'] for row in held]
        encoded = features(questions).to(device)
        lengths = torch.tensor([len(question) for question in questions], device=device)
        kinds = torch.tensor([tuple(LABELS).index(row['kind']) for row in held], device=device)
        heads = planner(encoded, lengths, kinds)
        rows = []
        for index, row in enumerate(held):
            kind = row['kind']
            current = row['fields'] | row['changes']
            gold = (tuple(row['order']), row['join'], row['style'])
            guessed = decode_plan(tuple(head[index] for head in heads), kind)
            case = {
                'turns': [row['first'], row['update'], row['question']],
                'initial': row['fields'], 'changes': row['changes'], 'current': current,
            }
            automatic_state, writes = write_metrics(locator, tokenizer, case)
            state_map = {'oracle': current, 'automatic': writes['final']}
            plan_map = {'gold': gold, 'predicted': guessed}
            output_row = {
                'kind': kind, 'question': row['question'], 'expected': row['expected'],
                'gold_plan': gold, 'predicted_plan': guessed, 'writes': writes,
                'programmatic': {
                    f'{state_name}_{plan_name}': plan_text(tokenizer, kind, fields, plan)
                    for state_name, fields in state_map.items()
                    for plan_name, plan in plan_map.items()
                },
                'native': {},
            }
            baseline = native_generate(decoder, tokenizer, row['question'], None, 0.)
            output_row['native']['baseline'] = {
                **baseline,
                **output_metrics(baseline['answer'], baseline['ended'], kind, current, gold, row['expected']),
            }
            for state_name, fields in state_map.items():
                for plan_name, plan in plan_map.items():
                    name = f'{state_name}_{plan_name}'
                    text = plan_text(tokenizer, kind, fields, plan)
                    generated = native_generate(decoder, tokenizer, row['question'], text, args.weight)
                    output_row['native'][name] = {
                        **generated,
                        **output_metrics(generated['answer'], generated['ended'], kind, fields, plan,
                                         row['expected']),
                    }
            rows.append(output_row)
            print(index + 1, kind, output_row['native']['baseline']['answer'], flush=True)

        dates = verify_date_isolation(
            runtime, WorkingTurnReader(writer_payload['hidden']).to(device), tokenizer,
            state_from(held[0]['fields'], tokenizer),
        )
        ordinary = ordinary_audit(decoder, tokenizer)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Native read-plan audit changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Native read-plan audit changed an input')

    conditions = ('baseline', 'oracle_gold', 'oracle_predicted',
                  'automatic_gold', 'automatic_predicted')
    scores = {name: summarize([row['native'][name if name == 'baseline' else
                                          f'{name.split("_")[0]}_{name.split("_")[1]}']
                               for row in rows]) for name in conditions}
    write(args.output / 'result.json', {
        'format': 'spora-native-read-plan-v1', 'weight': args.weight,
        'scores': scores, 'cases': rows, 'date': dates, 'ordinary': ordinary,
        'digest': digest, 'planner_sha256': sha(args.planner),
        'inputs': {str(path): value for path, value in sources.items()},
        'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    })
    atomic_save({'format': 'spora-native-read-plan-v1', 'weight': args.weight,
                 'parent_sha256': sources[args.checkpoint], 'planner_sha256': sha(args.planner)},
                args.output / 'audit.pt')
    print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'planner', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--weight', type=float, default=8.)
    options = parser.parse_args()
    if not 0. <= options.weight <= 64.:
        parser.error('--weight must be in [0, 64]')
    main(options)
