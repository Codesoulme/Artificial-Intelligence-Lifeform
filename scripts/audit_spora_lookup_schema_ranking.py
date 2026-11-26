"""Compare learned field labels with description-conditioned address scores."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_lookup_field_focus as focus_audit
from scripts import audit_spora_switch_address_ranking as address_audit
from scripts.audit_spora_lookup_field_focus import dense, plan
from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


FIELD_DESCRIPTIONS = {
    'drinks': '供人喝的液体，记录约定或选定的饮料种类。',
    'places': '人与人约定会合的地理位置，记录碰面的场所。',
    'items': '要携带、交付或备齐的实物，记录约定所需的物件。',
    'books': '指定阅读、借阅或查找的出版物，记录作品的名称。',
}
FIELD_CUTS = (.25, .4, .5, .65, .8, .9, .95, .99, 1.01)


class DescriptionPointer(dense.JointPointer):
    def __init__(self, hidden, descriptions):
        super().__init__(hidden)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64, bias=False))
        self.register_buffer('descriptions', descriptions, persistent=False)

    def forward(self, global_state, evidence, allowed):
        global_state = self.shared(global_state)
        owners = F.gelu(self.local(evidence) + global_state[:, None] + self.owner_position)
        fields = self.output(self.descriptions)
        scores = (owners @ fields.T / fields.shape[-1]**.5).flatten(1)
        return torch.cat((scores.masked_fill(~allowed, -torch.inf), self.none(global_state)), dim=-1)


def best_cutoff(rows, evidence, pointer, intent):
    def utility(cutoff):
        counts = focus_audit.score_single(rows, evidence, pointer, intent, cutoff)['counts']
        errors = counts['wrong'] + counts['unnecessary'] + counts['substitute']
        return counts['correct'] - 2 * errors, -errors, -cutoff

    return max(FIELD_CUTS, key=utility)


def counts_by_field(outcome):
    return {field: dict(total=len(rows), correct=sum(row['candidate'] == row['expected'] for row in rows),
                        wrong=sum(row['candidate'] is not None and row['candidate'] != row['expected'] for row in rows),
                        first=sum(row['unmasked'] == row['expected'] for row in rows))
            for field in plan.FIELDS_ORDER
            if (rows := [row for row in outcome['rows'] if row['expected'] and row['expected'][1] == field])}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    baseline = json.loads((args.baseline / 'result.json').read_text(encoding='utf-8'))
    previous = json.loads((args.focus / 'result.json').read_text(encoding='utf-8'))
    field_path, intent_path = (args.baseline / f'{name}.pt' for name in ('field', 'intent'))
    focus_path = args.focus / 'independent.pt'
    backbone = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in (field_path, intent_path, focus_path, *backbone)}
    previous_sources = {Path(path).resolve(): digest for path, digest in previous['sources'].items()}
    expected = {path: previous_sources.get(path.resolve()) for path in sources}
    expected[focus_path] = previous['independent_sha256']
    if sources != expected:
        raise ValueError('The field and clause heads must match the evaluated recurrent backbone')

    train_owners = ('马澄文', '章其安', '窦明川', '戚时月')
    train_clauses = focus_audit.paired_cases(train_owners[2:], focus_audit.TRAIN)
    train = list(dict.fromkeys([
        *address_audit.training_cases(train_owners[:2], dense.ALIASES, address_audit.ADDRESS_FORMS),
        *address_audit.matched_cases(train_owners[:2], address_audit.MATCHED_TRAIN),
        *((parts[index], target) for parts, index, target in train_clauses),
    ]))
    cal_owners = ('庄临修', '乔文荃')
    held_owners = ('夏宁瑶', '郭回霜')
    cal = focus_audit.paired_cases(cal_owners, focus_audit.CALIBRATION)
    calibration = cal + focus_audit.no_single_cases(cal_owners, focus_audit.CALIBRATION)
    held = focus_audit.paired_cases(held_owners, focus_audit.HELD)
    transfer = (focus_audit.paired_cases(held_owners, focus_audit.TRANSFER)
                + focus_audit.no_single_cases(held_owners, focus_audit.TRANSFER))
    deferred = [(tuple(part.format(owner=owner) for part in parts), -1, None)
                for owner in held_owners for parts in focus_audit.DEFERRED]
    suites = {'calibration': (calibration, cal_owners), 'held': (held, held_owners),
              'transfer': (transfer, held_owners), 'deferred': (deferred, held_owners)}
    normalized = {plan.anonymize(text, train_owners)[0] for text, _ in train}
    for rows, owners in suites.values():
        clauses = {plan.anonymize(part, owners)[0] for parts, _, _ in rows for part in parts}
        if normalized & clauses:
            raise ValueError('Training and evaluation clauses overlap')

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        width = model.config.hidden_size
        intent = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1)).eval()
        intent_state = torch.load(intent_path, map_location='cpu', weights_only=True)
        intent.load_state_dict(intent_state, strict=True)
        focus = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1)).eval()
        focus.load_state_dict(torch.load(focus_path, map_location='cpu', weights_only=True), strict=True)
        old = dense.JointPointer(width).eval()
        old.load_state_dict(torch.load(field_path, map_location='cpu', weights_only=True), strict=True)
        with torch.inference_mode():
            descriptions, _, _ = dense.align(extract, [FIELD_DESCRIPTIONS[field] for field in plan.FIELDS_ORDER], ())
        descriptions = descriptions.clone()
        train_inputs = dense.align(extract, [text for text, _ in train], train_owners)
        label, _ = address_audit.fit_split(train, train_inputs, fixed_intent=intent_state)
        schema, _ = address_audit.fit_split(train, train_inputs, fixed_intent=intent_state,
                                           pointer_factory=lambda width: DescriptionPointer(width, descriptions))
        models = {'previous': old, 'label': label, 'description': schema}
        digests = {}
        for name in ('label', 'description'):
            path = args.output / f'{name}.pt'
            atomic_save(models[name].state_dict(), path)
            models[name].load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            digests[name] = base.sha(path)
        evidence = {name: focus_audit.single_evidence(extract, rows, focus, owners, previous['independent_cutoff'])
                    for name, (rows, owners) in suites.items()}
        cutoffs = {name: best_cutoff(calibration, evidence['calibration'], pointer, intent)
                   for name, pointer in models.items() if name != 'previous'}
        cutoffs['previous'] = baseline['separate_training']['cutoffs'][1]
        results = {}
        for suite, (rows, _) in suites.items():
            results[suite] = {}
            for name, pointer in models.items():
                outcome = focus_audit.score_single(rows, evidence[suite], pointer, intent, cutoffs[name])
                outcome['fields'] = counts_by_field(outcome)
                results[suite][name] = outcome
                print(suite, name, cutoffs[name], outcome['intent_correct'], outcome['counts'], outcome['fields'], flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model input changed')
    store_json(args.output / 'result.json', dict(results=results, cutoffs=cutoffs, training=len(train),
                                               descriptions=FIELD_DESCRIPTIONS, checkpoints=digests,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('baseline', type=Path)
    cli.add_argument('focus', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
