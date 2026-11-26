"""Compare lookup intent readouts before and after the recurrent attention layers."""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn

from scripts import audit_spora_lookup_field_focus as audit
from scripts.audit_spora_lookup_field_focus import dense, plan
from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


TRAIN = (
    ('你去确认{cue}', '我去确认{cue}'),
    ('这件事麻烦你核对{cue}', '这件事我稍后核对{cue}'),
    ('帮忙翻出{cue}', '先不用翻出{cue}'),
    ('回顾一下保存过的{cue}', '我想自己回顾保存过的{cue}'),
    ('能把记下的{cue}告诉我吗', '先别把记下的{cue}告诉我'),
    ('给我念出{cue}', '保留原话「给我念出{cue}」'),
)
CAL = (
    ('由你核实{cue}', '由我核实{cue}'),
    ('报一下留档的{cue}', '留档的{cue}先别报'),
    ('对一下{cue}叫什么', '我会晚些对一下{cue}叫什么'),
)
HELD = (
    ('劳你跟我对一对{cue}', '这回我来跟本人对一对{cue}'),
    ('把之前留底的{cue}讲给我听', '我改日自己弄清之前留底的{cue}'),
    ('说一下{cue}的名称', '目前不必说{cue}的名称'),
    ('把{cue}的名字报出来', '改写「把{cue}的名字报出来」这句话'),
)
CUES = (
    ('{owner}说要喝的那杯', '{owner}约好等人的地方', '{owner}要我带去的物件', '{owner}打算读的书'),
    ('{owner}点的喝的东西', '{owner}定下来碰头的位置', '{owner}嘱咐带好的用具', '{owner}借来的读物'),
    ('{owner}那天预订的饮料', '{owner}提到的会合处', '{owner}让人捎过去的东西', '{owner}选来翻阅的那本'),
)
STATES = ((False, False), (True, False), (False, True), (True, True))


def matched(families, owners, cues):
    examples = []
    for family, (ask, skip) in enumerate(families):
        for first in range(len(cues)):
            second = (first + 1 + family % (len(cues) - 1)) % len(cues)
            names = owners if first % 2 == 0 else owners[::-1]
            topics = tuple(cues[field].format(owner=name) for field, name in zip((first, second), names, strict=True))
            addresses = tuple(MemoryAddress(name, plan.FIELDS_ORDER[field])
                              for field, name in zip((first, second), names, strict=True))
            for reads in STATES:
                parts = tuple((ask if read else skip).format(cue=cue) for read, cue in zip(reads, topics, strict=True))
                selected = reads.index(True) if sum(reads) == 1 else -1
                address = addresses[selected] if selected >= 0 else None
                examples.append(((parts, selected, address), reads, family))
    return examples


def by_request_type(examples, outcome, cutoff):
    counts = {name: Counter() for name in ('zero', 'single', 'double')}
    family_counts = {}
    for (_, reads, family), scores, selected, row in zip(
            examples, outcome['clause_confidence'], outcome['selected'], outcome['rows'], strict=True):
        bucket = counts[('zero', 'single', 'double')[sum(reads)]]
        detected = tuple(score >= cutoff for score in scores)
        correct = sum(reads) == 1 and row['candidate'] == row['expected']
        bucket.update(total=1, exact_clauses=detected == reads, proposed=selected >= 0, correct_address=correct)
        family_counts.setdefault(family, Counter()).update(total=1, exact_clauses=detected == reads,
                                                            correct_address=correct)
    return dict(requests=counts, families=family_counts)


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    previous = json.loads((args.focus / 'result.json').read_text(encoding='utf-8'))
    schema = json.loads((args.schema / 'result.json').read_text(encoding='utf-8'))
    focus_path = args.focus / 'independent.pt'
    field_path = args.schema / 'label.pt'
    intent_path = args.baseline / 'intent.pt'
    recurrent = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in (focus_path, field_path, intent_path, *recurrent)}
    expected = {Path(path).resolve(): digest for path, digest in schema['sources'].items()}
    expected[focus_path.resolve()] = previous['independent_sha256']
    expected[field_path.resolve()] = schema['checkpoints']['label']
    if any(expected.get(path.resolve()) != digest for path, digest in sources.items()):
        raise ValueError('Model inputs do not match the preceding experiment')

    train_names = ('窦明川', '戚时月')
    cal_names = ('储映萱', '章林霄')
    held_names = ('何知彦', '沈琢云')
    extra = matched(TRAIN, train_names, CUES[0])
    original = audit.paired_cases(train_names, audit.TRAIN)
    train = original + [row for row, _, _ in extra]
    targets = torch.tensor([tuple(index == slot for slot in range(2)) for _, index, _ in original]
                           + [reads for _, reads, _ in extra])
    calibration = matched(CAL, cal_names, CUES[1])
    held = matched(HELD, held_names, CUES[2])
    prior_names = ('夏宁瑶', '郭回霜')
    deferred = [(tuple(part.format(owner=owner) for part in parts), -1, None)
                for owner, parts in product(prior_names, audit.DEFERRED)]
    suites = {
        'calibration': ([row for row, _, _ in calibration], cal_names),
        'held': ([row for row, _, _ in held], held_names),
        'prior_held': (audit.paired_cases(prior_names, audit.HELD), prior_names),
        'prior_transfer': (audit.paired_cases(prior_names, audit.TRANSFER)
                           + audit.no_single_cases(prior_names, audit.TRANSFER), prior_names),
        'prior_deferred': (deferred, prior_names),
    }
    normalized = {plan.anonymize(part, train_names)[0] for parts, _, _ in train for part in parts}
    for rows, owners in suites.values():
        clauses = {plan.anonymize(part, owners)[0] for parts, _, _ in rows for part in parts}
        if normalized & clauses:
            raise ValueError('Training and evaluation contain the same anonymized clause')

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    checkpoints = {}
    with local_model(args.recurrent) as model:
        width = model.config.hidden_size
        make_head = lambda: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))
        field = dense.JointPointer(width).eval()
        field.load_state_dict(torch.load(field_path, map_location='cpu', weights_only=True), strict=True)
        intent = make_head().eval()
        intent.load_state_dict(torch.load(intent_path, map_location='cpu', weights_only=True), strict=True)
        decoder = SimpleNamespace(base=model)
        field_extract = FrozenTurnFeatures(decoder, tokenizer, depth=plan.DEPTH)
        for depth in (plan.DEPTH, 23):
            extract = (field_extract if depth == plan.DEPTH else
                       FrozenTurnFeatures(decoder, tokenizer, depth=depth, single_sequence=True))
            trained = audit.fit_focus(audit.encode_clauses(extract, train, train_names), train,
                                      independent=True, clause_reads=targets)
            path = args.output / f'intent-depth-{depth}.pt'
            atomic_save(trained.state_dict(), path)
            trained.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            checkpoints[depth] = base.sha(path)
            heads = {f'depth_{depth}': trained}
            if depth == plan.DEPTH:
                heads['previous'] = make_head().eval()
                heads['previous'].load_state_dict(torch.load(focus_path, map_location='cpu', weights_only=True), strict=True)
            features = {name: audit.encode_clauses(extract, rows, owners) for name, (rows, owners) in suites.items()}
            for name, head in heads.items():
                with torch.inference_mode():
                    confidence = {key: head(values).squeeze(-1).sigmoid() for key, values in features.items()}
                cutoff = (previous['independent_cutoff'] if name == 'previous' else
                          audit.calibration_cut(confidence['calibration'], suites['calibration'][0]))
                outcomes = {}
                for suite, (rows, owners) in suites.items():
                    evidence = audit.single_evidence(field_extract, rows, head, owners, cutoff,
                                                     clause_confidence=confidence[suite])
                    outcome = audit.score_single(rows, evidence, field, intent, schema['cutoffs']['label'])
                    if suite in ('held', 'calibration'):
                        outcome.update(by_request_type(held if suite == 'held' else calibration, outcome, cutoff))
                    outcomes[suite] = outcome
                    print(name, cutoff, suite, outcome['intent_correct'], outcome['counts'], outcome.get('requests'), flush=True)
                results[name] = dict(cutoff=cutoff, outcomes=outcomes)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model input changed')
    store_json(args.output / 'result.json', dict(results=results, training=len(train), checkpoints=checkpoints,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('baseline', type=Path)
    cli.add_argument('focus', type=Path)
    cli.add_argument('schema', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
