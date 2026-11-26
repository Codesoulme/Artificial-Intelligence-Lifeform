"""Train and audit working-state writes and native answer reads on frozen DeepSeek."""

import argparse
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import encode
from scripts.audit_spora_chat_backbones import reply
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_composition import DATA as NATURAL, parse_answer, settings_for, tile_logits
from scripts.train_spora_dialogue_working_state import (DATA, MARGIN, SEED, WRITER_WIDTH, dialogue_data,
                                                       fit_writer, writer_results)
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader, WorkingTurnState
from train_spora_query_key_dialogue import answer_request


FEATURE_DEPTH = 12
FORMAT = 'spora-working-turn-experiment-v1'


class FrozenTurnFeatures:
    def __init__(self, decoder, tokenizer, *, depth=FEATURE_DEPTH, single_sequence=False):
        self.model = decoder.base.model
        self.device = decoder.base.device
        self.tokenizer = tokenizer
        if not 1 <= depth <= len(self.model.layers):
            raise ValueError('Feature depth must name an existing decoder layer')
        self.depth = depth
        self.single_sequence = single_sequence

    @torch.no_grad()
    def __call__(self, texts):
        width = max(map(len, texts))
        if self.single_sequence and len(texts) > 1:
            return torch.cat([torch.nn.functional.pad(self([text]), (0, 0, 0, width - len(text)))
                              for text in texts])
        encoded = self.tokenizer(list(texts), padding=True, add_special_tokens=False,
                                 return_tensors='pt', return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')
        # Consecutive byte tokens may cover the same character; prefer the last.
        positions = torch.zeros(len(texts), width, dtype=torch.long)
        for i, row in enumerate(offsets.tolist()):
            for token, (begin, end) in enumerate(row):
                positions[i, begin:end] = token
        layers = self.model.layers
        self.model.layers = nn.ModuleList(layers[:self.depth])
        try:
            hidden = []
            for begin in range(0, len(texts), 12):
                subset = slice(begin, begin + 12)
                local = {key: value[subset].to(self.device) for key, value in encoded.items()}
                output = self.model(**local, use_cache=False).last_hidden_state
                indices = positions[subset].to(self.device)
                gathered = output.gather(1, indices.unsqueeze(-1).expand(-1, -1, output.shape[-1]))
                hidden.append(gathered.cpu())
            return torch.cat(hidden)
        finally:
            self.model.layers = layers


class SemanticLocator:
    def __init__(self, writer, features):
        self.writer = writer
        self.features = features

    def __call__(self, text, margin=MARGIN):
        context = self.features([text]).to(self.writer.embedding.weight.device)
        return self.writer.write(text, margin, features=context)


def state_from(fields, tokenizer, previous=None):
    state = WorkingTurnState() if previous is None else previous
    return state.update({field: tuple(tokenizer.encode(value, add_special_tokens=False))
                         for field, value in fields.items()})


def rollout_states(locator, tokenizer, turns):
    state = WorkingTurnState()
    rows = []
    for text in turns:
        changes = locator(text)
        state = state_from(changes, tokenizer, previous=state)
        rows.append(changes)
    return state, rows


def train_examples(rows, tokenizer, *, contrast):
    examples = []
    for case in rows:
        first = '好的，我记下了。'
        second = '明白，我会按新的要求处理。'
        messages = [{'role': role, 'content': text} for role, text in
                    (('user', case['first'][0]), ('assistant', first), ('user', case['update'][0]),
                     ('assistant', second), ('user', case['question']))]
        contexts = [messages, messages[-1:]] if contrast else [messages]
        for context in contexts:
            prompt = encode(tokenizer, context)
            versions = [('current', 'answer')]
            if contrast:
                versions.append(('initial', 'initial_answer'))
            paired = []
            for state_key, answer_key in versions:
                ids, labels, _ = answer_request(tokenizer, prompt, case[answer_key], {})
                paired.append((state_from(case[state_key], tokenizer), ids, labels))
            examples.append(paired)
    return examples


def fit_reader(decoder, reader, tokenizer, cases, steps, *, contrast=False):
    rng = random.Random(SEED)
    examples = train_examples(cases, tokenizer, contrast=contrast)
    params = [p for name, p in reader.named_parameters() if name not in ('copy_scale', 'order_scale')]
    scale_params = [reader.copy_scale] if reader.order_scale is None else [reader.copy_scale, reader.order_scale]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': .001},
                                   {'params': scale_params, 'lr': .01}], weight_decay=0)
    losses, trace = [], []
    pair_width = 2 if contrast else 1
    pair = None
    for step in range(steps):
        # Each iteration projects a different subset of response positions.
        arm = step % pair_width
        if arm == 0:
            pair = rng.choice(examples)
        state, ids, labels = pair[arm]
        groups = (len(labels) + 3) // 4
        indices = tuple(range(rng.randrange(groups), len(labels), groups))
        pointer = torch.tensor([labels[i] for i in indices], device=decoder.base.device) if contrast else None
        with reader.active(decoder, state, targets=pointer, prompt_tokens=len(ids) - len(labels) + 1) as guidance:
            logits, target = tile_logits(decoder, (ids, labels, indices))
        loss = F.cross_entropy(logits, target)
        if guidance:
            loss = loss + .55 * guidance[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            item = {'step': step + 1, 'loss': sum(losses) / len(losses),
                    'copy_scale': float(12. * reader.copy_scale.detach().tanh())}
            if reader.order_scale is not None:
                item['order_scale'] = float(8. * reader.order_scale.detach().tanh())
            trace.append(item)
            print('Working-state reader', item, flush=True)
            losses.clear()
    reader.eval()
    return trace


def measure(decoder, reader, tokenizer, settings, history, question, state):
    with reader.active(decoder, state):
        return reply(decoder, tokenizer, history, question, **settings)


def score_case(case, generated):
    text = generated['answer']
    present = {field: value in text for field, value in case['current'].items()}
    stale = any(case['initial'][field] in text and case['initial'][field] not in value
                for field, value in case['changes'].items())
    form = True
    if case['kind'] == 'parcel':
        fields = [re.escape(case['current'][field]) for field in ('person', 'quantity', 'place')]
        form = re.fullmatch(r'\s*[/／]\s*'.join(fields) + r'[。.]?', text) is not None
    return {'present': present, 'stale': stale, 'form': form,
            'complete': all(present.values()) and not stale and form and generated['eos']}


def controlled(decoder, reader, tokenizer, locator, held, *, question_only=False):
    settings = settings_for(decoder, tokenizer, 48)
    results = []
    for case in held:
        turns = [case['first'][0], case['update'][0], case['question']]
        state, writes = rollout_states(locator, tokenizer, turns)
        history = []
        for user in turns[:2]:
            generated = reply(decoder, tokenizer, history, user, **settings)
            history.extend(({'role': 'user', 'content': user}, {'role': 'assistant', 'content': generated['answer']}))
        inputs = {'off': WorkingTurnState(), 'predicted': state,
                  'current_oracle': state_from(case['current'], tokenizer),
                  'initial_oracle': state_from(case['initial'], tokenizer)}
        versions = {}
        for mode, version in inputs.items():
            generated = measure(decoder, reader, tokenizer, settings, history, case['question'], version)
            versions[mode] = {'generated': generated, **score_case(case, generated)}
        if question_only:
            for mode, version in (('question_predicted', state), ('question_oracle', inputs['current_oracle'])):
                generated = measure(decoder, reader, tokenizer, settings, [], case['question'], version)
                versions[mode] = {'generated': generated, **score_case(case, generated)}
        new_answer = versions['current_oracle']['generated']['answer']
        old_answer = versions['initial_oracle']['generated']['answer']
        switched = all(value in new_answer and case['initial'][field] in old_answer
                       and (value in case['initial'][field] or value not in old_answer)
                       and (case['initial'][field] in value or case['initial'][field] not in new_answer)
                       for field, value in case['changes'].items())
        results.append({'kind': case['kind'], 'turns': turns, 'history': history, 'writes': writes,
                        'switched': switched,
                        'expected': case['current'], 'changes': case['changes'], 'versions': versions})
    print('Controlled complete', {mode: sum(row['versions'][mode]['complete'] for row in results)
                                  for mode in results[0]['versions']}, flush=True)
    return results


def natural_cases(decoder, reader, tokenizer, locator, before):
    cases = json.loads(NATURAL.read_text(encoding='utf-8'))['evaluation']
    settings = settings_for(decoder, tokenizer, 64)
    rows = {}
    for case in cases:
        key = case['name']
        prior = before[key]['turns']
        if [row['question'] for row in prior] != case['turns']:
            raise ValueError('Natural evaluation questions changed')
        state, writes = rollout_states(locator, tokenizer, case['turns'])
        messages = [{'role': role, 'content': row[field]} for row in prior[:2]
                    for role, field in (('user', 'question'), ('assistant', 'answer'))]
        generated = measure(decoder, reader, tokenizer, settings, messages, case['turns'][-1], state)
        baseline = prior[-1]
        scores = [parse_answer(baseline['answer'], case), parse_answer(generated['answer'], case)]
        rows[key] = {'writes': writes, 'old': {'generated': baseline, 'score': scores[0]},
                     'working': {'generated': generated, 'score': scores[1]}}
    print('Natural complete', {arm: sum(case[arm]['score']['complete'] and case[arm]['generated']['eos']
                                        for case in rows.values()) for arm in ('old', 'working')}, flush=True)
    return rows


def verify_date_isolation(runtime, reader, tokenizer, state):
    from scripts.audit_spora_mixed_calendar_chat import SCENES
    from scripts.spora_calendar_history_chat import calendar_turn

    def probe():
        rows = []
        for turns in SCENES.values():
            focus = None
            for question, expected, _ in turns:
                if expected != 'ordinary_chat':
                    result, focus = calendar_turn(runtime.decoder, tokenizer, runtime.organism, runtime.state,
                                                  runtime.observation_calendar, question, focus)
                    rows.append(result)
        return rows

    normal = probe()
    with reader.active(runtime.decoder, state):
        affected = probe()
    if normal != affected:
        raise ValueError('Working-state hooks changed dated memory responses')
    return normal


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    protected = {path: sha(path) for path in (args.checkpoint, args.database, args.before, DATA, NATURAL)}
    if args.writer_from is not None:
        protected[args.writer_from] = sha(args.writer_from)
    config = json.loads(DATA.read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    before = json.loads(args.before.read_text(encoding='utf-8'))
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        digest = frozen_digest(runtime.organism, runtime.state)
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        hidden = decoder.base.config.hidden_size
        features = FrozenTurnFeatures(decoder, tokenizer)
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=hidden).to(device)
        train = dialogue_data(config, held=False, count=128)
        held = dialogue_data(config, held=True, count=8)
        if args.writer_from is None:
            trace = fit_writer(writer, train, config['no_write'], args.epochs, device, features=features)
        else:
            source = torch.load(args.writer_from, map_location='cpu', weights_only=True)
            if (source['format'] != FORMAT or source['parent_sha256'] != protected[args.checkpoint]
                    or source['fields'] != FIELDS or source['feature_depth'] != FEATURE_DEPTH):
                raise ValueError('Existing writer was trained for a different decoder')
            writer.load_state_dict(source['writer'])
            trace = []
        writer.eval()
        locator = SemanticLocator(writer, features)
        writes = writer_results(writer, held, locator)
        print('Contextual writer exact', sum(row['complete'] for row in writes), '/', len(writes), flush=True)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'writes.json', {'cases': writes, 'trace': trace})
        torch.cuda.empty_cache()
        reader = WorkingTurnReader(hidden).to(device)
        reader_trace = fit_reader(decoder, reader, tokenizer, train, args.steps, contrast=args.contrast)
        adapter = args.output / 'working-state.pt'
        atomic_save({'format': FORMAT, 'parent_sha256': protected[args.checkpoint], 'fields': FIELDS,
                     'hidden': hidden, 'feature_depth': FEATURE_DEPTH, 'writer_width': WRITER_WIDTH,
                     'writer': writer.cpu().state_dict(), 'reader': reader.cpu().state_dict()}, adapter)
        writer.to(device)
        reader.to(device)
        controlled_rows = controlled(decoder, reader, tokenizer, locator,
                                     [case for local, case in enumerate(held) if local % 8 < 4],
                                     question_only=args.contrast)
        natural = natural_cases(decoder, reader, tokenizer, locator, before)
        date = verify_date_isolation(runtime, reader, tokenizer, state_from(held[0]['current'], tokenizer))
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Working state changed the durable neural individual')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('Working-state audit changed an input')
    write(args.output / 'result.json', {'writes': writes, 'reader_trace': reader_trace, 'contrast': args.contrast,
                                        'controlled': controlled_rows, 'natural': natural, 'date': date,
                                        'digest': digest, 'adapter_sha256': sha(adapter),
                                        'sources': {str(path): digest for path, digest in protected.items()},
                                        'seconds': time.perf_counter() - started,
                                        'peak_bytes': torch.cuda.max_memory_allocated()})


def verify(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protected = {path: sha(path) for path in (args.checkpoint, args.database, args.before, args.adapter, args.verify,
                                             DATA, NATURAL)}
    expected = json.loads(args.verify.read_text(encoding='utf-8'))
    if (protected[args.adapter] != expected['adapter_sha256']
            or any(protected[path] not in expected['sources'].values()
                   for path in (args.checkpoint, args.database, args.before, DATA, NATURAL))):
        raise ValueError('Reopen requires the same source inputs')
    payload = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if (payload['format'] != FORMAT or payload['parent_sha256'] != protected[args.checkpoint]
            or payload['fields'] != FIELDS or payload['feature_depth'] != FEATURE_DEPTH
            or payload['writer_width'] != WRITER_WIDTH):
        raise ValueError('Working-state adapter metadata does not match the decoder')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    before = json.loads(args.before.read_text(encoding='utf-8'))
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        hidden = decoder.base.config.hidden_size
        if payload['hidden'] != hidden:
            raise ValueError('Working-state adapter width does not match the decoder')
        device = decoder.base.device
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=hidden)
        reader = WorkingTurnReader(hidden)
        writer.load_state_dict(payload['writer'])
        reader.load_state_dict(payload['reader'])
        writer.to(device).eval().requires_grad_(False)
        reader.to(device).eval().requires_grad_(False)
        locator = SemanticLocator(writer, FrozenTurnFeatures(decoder, tokenizer))
        actual = natural_cases(decoder, reader, tokenizer, locator, before)
        held = dialogue_data(json.loads(DATA.read_text(encoding='utf-8')), held=True, count=8)
        date = verify_date_isolation(runtime, reader, tokenizer, state_from(held[0]['current'], tokenizer))
        for rows in (actual, expected['natural']):
            for case in rows.values():
                for arm in ('old', 'working'):
                    case[arm]['generated'].pop('seconds')
        if (actual != expected['natural'] or date != expected['date']
                or frozen_digest(runtime.organism, runtime.state) != expected['digest']):
            raise ValueError('Working-state reload changed replies or dated memory')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('Working-state reload changed an input')
    write(args.output, {'natural_verified': len(actual), 'date_verified': len(date),
                        'adapter_sha256': protected[args.adapter]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'before', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--steps', type=int, default=192)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--writer-from', type=Path)
    parser.add_argument('--contrast', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1:
        parser.error('Training counts must be positive')
    if (args.adapter is None) != (args.verify is None):
        parser.error('--adapter and --verify are required together')
    if args.verify is None:
        main(args)
    else:
        verify(args)
