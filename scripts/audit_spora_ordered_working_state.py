"""Measure sequential copy, answer completion and stopping on held working states."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from string import Formatter
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, LogitsProcessor

from audit_spora_dialogue_working_state import (DATA, FEATURE_DEPTH, FORMAT, WRITER_WIDTH,
    FrozenTurnFeatures, SemanticLocator, rollout_states, state_from, verify_date_isolation)
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import _chat_inputs
from scripts.train_spora_dialogue_working_state import SEED, dialogue_data
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, SPAN_LIMIT, TurnValueWriter


POINTERS = len(FIELDS) * SPAN_LIMIT
WIDTH = 64


def answer_actions(template, fields, tokenizer, glue):
    table = {token: i for i, token in enumerate(glue, start=POINTERS)}
    actions = []
    for text, field, _, _ in Formatter().parse(template):
        actions.extend(table[token] for token in tokenizer.encode(text, add_special_tokens=False))
        if field is not None:
            length = len(tokenizer.encode(fields[field], add_special_tokens=False))
            offset = FIELDS.index(field) * SPAN_LIMIT
            actions.extend(range(offset, offset + length))
    actions.append(table[tokenizer.eos_token_id])
    return actions


def state_tokens(state, glue, device):
    slots = [list(tokens) + [-1] * (SPAN_LIMIT - len(tokens)) for tokens in state.slots]
    return torch.tensor([token for slot in slots for token in slot] + glue, dtype=torch.long, device=device)


class OrderedRead(nn.Module):
    """Autoregressive pointer over live slots and learned glue-token actions."""

    def __init__(self, glue_size):
        super().__init__()
        size = POINTERS + glue_size
        self.actions = nn.Embedding(size + 1, WIDTH)
        self.context = nn.Linear(POINTERS, WIDTH)
        self.recurrent = nn.GRU(2 * WIDTH, WIDTH, batch_first=True)
        self.output = nn.Linear(WIDTH, size)
        self.start = size

    def forward(self, previous, valid):
        context = self.context(valid[:, :POINTERS].float()).unsqueeze(1)
        inputs = torch.cat((self.actions(previous), context.expand(-1, previous.shape[1], -1)), dim=-1)
        hidden, _ = self.recurrent(inputs)
        return self.output(hidden).masked_fill(~valid[:, None, :], -torch.inf)


class OrderedScores(LogitsProcessor):
    def __init__(self, reader, tokens, prefix_length, weight, mix):
        self.reader, self.tokens = reader, tokens
        self.prefix_length = prefix_length
        self.weight = weight
        self.mix = mix
        self.actions = [reader.start]
        self.prior = None

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1 or input_ids.shape[1] != self.prefix_length + len(self.actions) - int(self.prior is None):
            raise ValueError('Ordered read requires single-sequence greedy generation')
        if self.prior is not None:
            matching = self.tokens == input_ids[0, -1]
            action = int(self.prior.masked_fill(~matching, -torch.inf).argmax()) if bool(matching.any()) else self.reader.start
            self.actions.append(action)
        previous = torch.tensor([self.actions], device=scores.device)
        valid = self.tokens.ge(0)
        self.prior = self.reader(previous, valid.unsqueeze(0))[0, -1].log_softmax(dim=-1)
        if self.mix is not None:
            copy = torch.zeros_like(scores, dtype=torch.float32)
            copy.scatter_add_(-1, self.tokens[valid].unsqueeze(0), self.prior[valid].exp().unsqueeze(0))
            language = scores.float().log_softmax(dim=-1) + math.log1p(-self.mix)
            return torch.logaddexp(language, copy.log() + math.log(self.mix))
        # A low-confidence slot prior leaves the ordinary decoder more influence.
        bonus = (self.weight + self.prior[valid]).clamp_min(0).to(scores.dtype)
        # The same token can occur at several addresses. Use max to avoid rewarding duplicates.
        return scores.scatter_reduce(-1, self.tokens[valid].unsqueeze(0), scores[:, self.tokens[valid]] + bonus,
                                     reduce='amax', include_self=True)


def train(reader, lengths, sequences, epochs, device):
    samples = torch.full((len(sequences), max(map(len, sequences))), -100, dtype=torch.long, device=device)
    for index, actions in enumerate(sequences):
        samples[index, :len(actions)] = torch.tensor(actions, device=device)
    previous = F.pad(samples[:, :-1], (1, 0), value=reader.start)
    previous = previous.masked_fill(previous < 0, reader.start)
    lengths = torch.stack(lengths)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=.003)
    trace = []
    for _ in range(epochs):
        for batch in torch.randperm(len(samples), device=device).split(96):
            logits = reader(previous[batch], lengths[batch])
            loss = F.cross_entropy(logits.flatten(0, 1), samples[batch].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        trace.append(float(loss.detach()))
    return trace


@torch.inference_mode()
def generate(decoder, tokenizer, question, state, reader, glue, weight, mix):
    inputs, _ = _chat_inputs(tokenizer, (), question, 384)
    inputs = inputs.to(decoder.base.device)
    length = inputs.input_ids.shape[1]
    kwargs = {}
    if state is not None and any(state.slots):
        tokens = state_tokens(state, glue, decoder.base.device)
        kwargs['logits_processor'] = [OrderedScores(reader, tokens, length, weight, mix)]
    out = decoder.generate(**inputs, use_cache=True, do_sample=False, max_new_tokens=40,
                           temperature=None, top_p=None, top_k=None, repetition_penalty=1.,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **kwargs)[0, length:]
    return {'answer': tokenizer.decode(out, skip_special_tokens=True).strip(),
            'ended': bool(len(out) and int(out[-1]) == tokenizer.eos_token_id), 'tokens': len(out)}


@torch.inference_mode()
def read_alone(reader, tokenizer, state, glue, device):
    tokens = state_tokens(state, glue, device)
    valid = tokens.ge(0).unsqueeze(0)
    previous = [reader.start]
    output = []
    for _ in range(40):
        actions = torch.tensor([previous], device=device)
        action = int(reader(actions, valid)[0, -1].argmax())
        token = int(tokens[action])
        output.append(token)
        if token == tokenizer.eos_token_id:
            break
        previous.append(action)
    return {'answer': tokenizer.decode(output, skip_special_tokens=True).strip(),
            'ended': output[-1] == tokenizer.eos_token_id, 'tokens': len(output)}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    started = time.perf_counter()
    protected = {path: sha(path) for path in (DATA, args.checkpoint, args.database, args.writer)}
    data = json.loads(DATA.read_text(encoding='utf-8'))
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'] != FORMAT or payload['parent_sha256'] != protected[args.checkpoint]
            or payload['fields'] != FIELDS or payload['feature_depth'] != FEATURE_DEPTH
            or payload['writer_width'] != WRITER_WIDTH):
        raise ValueError('Working-state writer and parent are incompatible')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    templates = {name: obj['answer'] for name, obj in data['kinds'].items()}
    glue = sorted({tokenizer.eos_token_id} | {token for template in templates.values()
                  for literal, _, _, _ in Formatter().parse(template)
                  for token in tokenizer.encode(literal, add_special_tokens=False)})
    device = 'cuda'
    torch.cuda.reset_peak_memory_stats()
    examples = dialogue_data(data, held=False, count=192)
    validity, actions = [], []
    for case in examples:
        for key in ('initial', 'current'):
            state = state_from(case[key], tokenizer)
            validity.append(state_tokens(state, glue, 'cpu').ge(0))
            actions.append(answer_actions(templates[case['kind']], case[key], tokenizer, glue))
    reader = OrderedRead(len(glue)).to(device)
    if args.read_from is None:
        losses = train(reader, [row.to(device) for row in validity], actions, args.epochs, device)
        print('Ordered reader training loss', losses[-1], flush=True)
    else:
        protected[args.read_from] = sha(args.read_from)
        saved = torch.load(args.read_from, map_location='cpu', weights_only=True)
        if (saved['format'], saved['glue'], saved['parent_sha256']) != (
            'spora-ordered-read-v1', glue, protected[args.checkpoint],
        ):
            raise ValueError('Ordered reader is incompatible with the current parent')
        reader.load_state_dict(saved['model'])
        losses = []
    reader.eval()
    torch.cuda.empty_cache()
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        digest = frozen_digest(runtime.organism, runtime.state)
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden'])
        writer.load_state_dict(payload['writer'])
        locator = SemanticLocator(writer.to(device).eval(), FrozenTurnFeatures(decoder, tokenizer))
        held = dialogue_data(data, held=True, count=4)
        rows = []
        for case in held:
            turns = [case['first'][0], case['update'][0], case['question']]
            current, initial = (state_from(case[key], tokenizer) for key in ('current', 'initial'))
            predicted, writes = rollout_states(locator, tokenizer, turns)
            modes = {'off': None, 'current': current, 'initial': initial, 'predicted': predicted}
            row = {'kind': case['kind'], 'question': case['question'], 'writes': writes}
            for mode, state in modes.items():
                result = generate(decoder, tokenizer, case['question'], state, reader, glue, args.weight, args.mix)
                expected = case['initial_answer'] if mode == 'initial' else case['answer']
                result.update(expected=expected, exact=result['answer'] == expected and result['ended'])
                row[mode] = result
                if state is not None and any(state.slots):
                    alone = read_alone(reader, tokenizer, state, glue, device)
                    alone.update(expected=expected, exact=alone['answer'] == expected and alone['ended'])
                    row[f'{mode}_alone'] = alone
            rows.append(row)
            print(case['kind'], {mode: row[mode]['exact'] for mode in modes}, flush=True)
        # OrderedScores is request-local and is not passed into dated generation.
        from spora.research_dialogue_working_state import WorkingTurnReader
        dates = verify_date_isolation(runtime, WorkingTurnReader(payload['hidden']).to(device), tokenizer, current)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Ordered read changed the durable state')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('Ordered read changed an input')
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_save({'format': 'spora-ordered-read-v1', 'glue': glue, 'model': reader.cpu().state_dict(),
                 'parent_sha256': protected[args.checkpoint]}, args.output / 'reader.pt')
    result = {'scores': dict(Counter(mode for row in rows for mode, value in row.items()
                                      if isinstance(value, dict) and value.get('exact'))),
              'rows': rows, 'date': dates, 'losses': losses, 'seconds': time.perf_counter() - started,
              'weight': args.weight,
              'mix': args.mix,
              'peak_bytes': torch.cuda.max_memory_allocated(), 'digest': digest,
              'sources': {str(path): value for path, value in protected.items()}}
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--read-from', type=Path)
    parser.add_argument('--weight', type=float, default=16.)
    parser.add_argument('--mix', type=float)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    if not 0 < options.weight <= 64:
        parser.error('--weight must be positive and bounded')
    if options.mix is not None and not 0 < options.mix < 1:
        parser.error('--mix must be between zero and one')
    main(options)
