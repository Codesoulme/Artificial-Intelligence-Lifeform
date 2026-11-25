"""Separate confirmation intent and candidate roles in SPORA time selection."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_active_time import ActiveTimeSelector, SEED, candidates, evaluate, features_for, fit
from scripts.audit_spora_active_time_composition import execute, requests
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_typed_composition import BINDING_PREFIX, SLOTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_active_time_cases import held_texts
from scripts.spora_time_authority_cases import ACTIVE, CONFIRMED, FRESH, training_pairs
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


MODES = ('old_selector', 'paired_selector', 'intent_and_role')


class IntentAndRole(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.intent = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 3))
        self.role = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, turn, features):
        return self.intent(turn.float()), self.role(features.float())

    def select(self, turn, features, valid):
        intent, role = self(turn, features)
        eligible = valid & (role.argmax(-1) == ACTIVE) & (intent.argmax(-1)[:, None] == CONFIRMED)
        scores = role.log_softmax(-1)[..., ACTIVE].masked_fill(~eligible, -torch.inf)
        absent = torch.zeros(len(turn)).masked_fill(eligible.any(-1), -torch.inf)
        return torch.cat((absent[:, None], scores), dim=1)


def turn_features(extract, texts):
    vectors = []
    inputs = [text + '\n这段话是在确定这次安排、表示尚未确定，还是在做别的事？' for text in texts]
    for start in range(0, len(inputs), 24):
        batch = inputs[start:start + 24]
        hidden = extract(batch)
        vectors.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(batch))
    return torch.stack(vectors)


def views(extract, cases):
    texts = [case.text for case in cases]
    matches, features, valid = features_for(extract, texts)
    return matches, features, valid, turn_features(extract, texts)


def fit_roles(head, train, matches, features, valid, turn):
    intents = torch.tensor([row.intent for row in train])
    roles = torch.zeros_like(valid, dtype=torch.long)
    for i, (row, spans) in enumerate(zip(train, matches, strict=True)):
        expected = dict(row.roles)
        if set(expected) != {span.group() for span in spans}:
            raise ValueError(f'Candidate role annotation does not cover the request: {row.example.text}')
        for j, span in enumerate(spans):
            roles[i, j] = expected[span.group()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for epoch in range(12):
        losses = []
        for batch in torch.randperm(len(train), generator=rng).split(24):
            intent, role = head(turn[batch], features[batch])
            active = valid[batch]
            local = F.cross_entropy(role[active], roles[batch][active], reduction='sum')
            loss = F.cross_entropy(intent, intents[batch]) + local / active.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('roles', epoch + 1, sum(losses) / len(losses), flush=True)


def classify_all(heads, cases, matches, features, valid, turn):
    outputs = {}
    for mode, head in heads.items():
        scorer = head if mode != 'intent_and_role' else lambda f, v: head.select(turn, f, v)
        rows, _ = evaluate(scorer, cases, matches, features, valid)
        outputs[mode] = rows
    return outputs


def count(rows):
    return Counter(cases=len(rows), correct=sum(row['selected'] == row['expected'] for row in rows),
                   false_accept=sum(row['expected'] is None and row['selected'] is not None for row in rows),
                   true_accept=sum(row['expected'] is not None and row['selected'] == row['expected'] for row in rows))


def run_composition(args, model, tokenizer, selections):
    path = args.records / 'versions.sqlite3'
    records = json.loads((args.records / 'result.json').read_text(encoding='utf-8'))
    typed = json.loads((args.typed / 'result.json').read_text(encoding='utf-8'))
    reader = args.typed / 'typed_time.pt'
    if digest(path) != records['database_sha256'] or digest(reader) != typed['head_sha256']['typed_time']:
        raise ValueError('Expected the audited frozen readout and version state')
    writer = next(hash_value for file, hash_value in records['sources'].items() if Path(file).name == 'adapters.pt')
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    state = torch.load(reader, map_location='cpu', weights_only=True)
    head = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=len(SLOTS))
    head.load_state_dict(state)
    with VersionStore(path, BINDING_PREFIX + writer + ':' + base.WEIGHT_SHA) as store:
        rows = {mode: list(execute(model, tokenizer, head.eval().to(model.device), codebook, store, selected, typed))
                for mode, selected in selections.items()}
    if digest(path) != records['database_sha256']:
        raise ValueError('A reply changed the stored version state')
    return rows


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('replay.json' if args.verify else 'result.json')
    if target.exists():
        raise FileExistsError(target)
    old = args.old / 'selector.pt'
    paths = (old, args.old / 'result.json', args.records / 'result.json', args.records / 'versions.sqlite3',
             args.typed / 'result.json', args.typed / 'typed_time.pt', base.MODEL / 'model.safetensors',
             base.MODEL / 'tokenizer.json', Path(__file__), Path(__file__).with_name('spora_time_authority_cases.py'))
    sources = {str(path): digest(path) for path in paths}
    prior = json.loads((args.old / 'result.json').read_text(encoding='utf-8'))
    if sources[str(old)] != prior['head_sha256'] or sources[str(base.MODEL / 'model.safetensors')] != base.WEIGHT_SHA:
        raise ValueError('Expected the audited selector and backbone')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    train = list(training_pairs())
    old_cases, fresh, joint = list(held_texts()), list(FRESH), [t.example for t in requests()]
    if {row.example.text for row in train} & {case.text for case in (*old_cases, *fresh, *joint)}:
        raise ValueError('An evaluation request appears verbatim in training')
    heads, checksums = {}, {}
    training_examples = [r.example for r in train]
    train_views = None if args.verify else views(extract, training_examples)
    for mode in MODES:
        torch.manual_seed(SEED)
        width = model.config.hidden_size
        head = IntentAndRole(width) if mode == 'intent_and_role' else ActiveTimeSelector(width)
        path = old if mode == 'old_selector' else args.output / (mode + '.pt')
        if args.verify or mode == 'old_selector':
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            if mode == 'intent_and_role':
                fit_roles(head, train, *train_views)
            else:
                fit(head, training_examples, *train_views[:3])
            atomic_save(head.state_dict(), path)
        heads[mode] = head.eval()
        checksums[mode] = digest(path)
    rows = {name: classify_all(heads, cases, *views(extract, cases))
            for name, cases in (('prior', old_cases), ('fresh', fresh), ('joint', joint))}
    if [row['selected'] for row in rows['prior']['old_selector']] != [row['selected'] for row in prior['rows']]:
        raise ValueError('The old selector did not reproduce the original choices')
    summary = {name: {mode: count(row) for mode, row in value.items()} for name, value in rows.items()}
    generation = run_composition(args, model, tokenizer, rows['joint'])
    if any(digest(path) != sources[str(path)] for path in paths):
        raise ValueError('A protected artifact changed')
    result = {'sources': sources, 'heads': checksums, 'train_count': len(train), 'rows': rows,
              'summary': summary, 'generation': generation, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        previous = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        for key in ('sources', 'heads', 'rows', 'summary', 'generation'):
            if json.loads(json.dumps(result[key])) != previous[key]:
                raise ValueError(f'Independent replay changed {key}')
    store_json(target, result)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'typed', 'records', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
