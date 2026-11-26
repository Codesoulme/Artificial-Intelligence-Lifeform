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
CONTEXT_FRAMES = (
    (
        ('先把{first}报给我', '刚才那条取消，现在帮我核对{second}', (False, True)),
        ('先把{first}报给我', '前一条照旧，也帮我核对{second}', (True, True)),
        ('麻烦翻看{first}', '我刚改主意了，两条都别查，{second}也先放着', (False, False)),
        ('麻烦翻看{first}', '只做刚才那条，{second}先放着', (True, False)),
    ),
    (
        ('帮我从留档里找{first}', '头一项不用做了，麻烦改成{second}', (False, True)),
        ('帮我从留档里找{first}', '头一项还要做，顺便也找下{second}', (True, True)),
        ('给我讲下{first}', '刚才那件收回，{second}同样留待以后', (False, False)),
        ('给我讲下{first}', '只讲我开头说的，{second}暂且缓一缓', (True, False)),
    ),
    (
        ('请你说说{first}', '前面那件划掉，换成查{second}', (False, True)),
        ('请你说说{first}', '原先拜托你的照常，再加上{second}', (True, True)),
        ('想请你对一下{first}', '两件一律先搁置，包括{second}', (False, False)),
        ('想请你对一下{first}', '只办前头提的那件，{second}往后排', (True, False)),
    ),
)


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


def context_cases(frames, owners, cues):
    examples = []
    for family, (first_frame, second_frame, reads) in enumerate(frames):
        for index, field in enumerate(plan.FIELDS_ORDER):
            other = (index + 1) % len(cues)
            names = owners if index % 2 == 0 else owners[::-1]
            values = dict(first=cues[index].format(owner=names[0]), second=cues[other].format(owner=names[1]))
            parts = first_frame.format(**values), second_frame.format(**values)
            addresses = MemoryAddress(names[0], field), MemoryAddress(names[1], plan.FIELDS_ORDER[other])
            selected = reads.index(True) if sum(reads) == 1 else -1
            address = addresses[selected] if selected >= 0 else None
            examples.append(((parts, selected, address), reads, f'context_{family}'))
    return examples


def intent_features(extract, rows, owners, *, contextual):
    local = audit.encode_clauses(extract, rows, owners)
    if not contextual:
        return local
    prompts = [plan.anonymize(question, owners)[0] for question, _ in audit.requests(rows)]
    snapshots = []
    for start in range(0, len(prompts), 12):
        batch = prompts[start:start + 12]
        hidden = extract(batch).float()
        snapshots.extend(hidden[index, len(prompt) - 1] for index, prompt in enumerate(batch))
    whole = torch.stack(snapshots)
    return torch.cat((local, whole[:, None].expand(-1, local.shape[1], -1)), dim=-1)


def contextual_cut(confidence, labels):
    expected = torch.tensor(labels)
    cuts = (.5, .65, .8, .9, .95, .98, .99, .995, 1.01)

    def utility(cutoff):
        detected = confidence.ge(cutoff)
        correct = detected.eq(expected).all(dim=1)
        zero_false = (~expected.any(dim=1) & detected.any(dim=1)).sum().item()
        return correct.sum().item() - zero_false, -zero_false, -cutoff

    return max(cuts, key=utility)


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
    measured = {'calibration': calibration, 'held': held}
    if args.contextual:
        train_context = context_cases(CONTEXT_FRAMES[0], train_names, CUES[0])
        calibration_context = context_cases(CONTEXT_FRAMES[1], cal_names, CUES[1])
        held_context = context_cases(CONTEXT_FRAMES[2], held_names, CUES[2])
        train.extend(row for row, _, _ in train_context)
        targets = torch.cat((targets, torch.tensor([reads for _, reads, _ in train_context])))
        measured.update(calibration=calibration + calibration_context, held_context=held_context)
        suites['calibration'] = ([row for row, _, _ in measured['calibration']], cal_names)
        suites['held_context'] = ([row for row, _, _ in held_context], held_names)
    normalized = {plan.anonymize(part, train_names)[0] for parts, _, _ in train for part in parts}
    for rows, owners in suites.values():
        clauses = {plan.anonymize(part, owners)[0] for parts, _, _ in rows for part in parts}
        if normalized & clauses:
            raise ValueError('Training and evaluation contain the same anonymized clause')

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    checkpoints = {}
    with local_model(args.recurrent, attention_mode=args.attention) as model:
        width = model.config.hidden_size
        make_head = lambda: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))
        field = dense.JointPointer(width).eval()
        field.load_state_dict(torch.load(field_path, map_location='cpu', weights_only=True), strict=True)
        intent = make_head().eval()
        intent.load_state_dict(torch.load(intent_path, map_location='cpu', weights_only=True), strict=True)
        decoder = SimpleNamespace(base=model)
        field_extract = FrozenTurnFeatures(decoder, tokenizer, depth=plan.DEPTH)
        arms = ((23, False), (23, True)) if args.contextual else ((plan.DEPTH, False), (23, False))
        for depth, contextual in arms:
            extract = (field_extract if depth == plan.DEPTH else
                       FrozenTurnFeatures(decoder, tokenizer, depth=depth, single_sequence=True))
            trained = audit.fit_focus(intent_features(extract, train, train_names, contextual=contextual), train,
                                      independent=True, clause_reads=targets)
            arm = 'whole_turn' if contextual else f'depth_{depth}'
            path = args.output / f'intent-{arm.replace("_", "-")}.pt'
            atomic_save(trained.state_dict(), path)
            trained.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            checkpoints[arm if args.contextual else depth] = base.sha(path)
            heads = {arm: trained}
            if depth == plan.DEPTH:
                heads['previous'] = make_head().eval()
                heads['previous'].load_state_dict(torch.load(focus_path, map_location='cpu', weights_only=True), strict=True)
            features = {name: intent_features(extract, rows, owners, contextual=contextual)
                        for name, (rows, owners) in suites.items()}
            for name, head in heads.items():
                with torch.inference_mode():
                    confidence = {key: head(values).squeeze(-1).sigmoid() for key, values in features.items()}
                if name == 'previous':
                    cutoff = previous['independent_cutoff']
                elif args.contextual:
                    cutoff = contextual_cut(confidence['calibration'], [reads for _, reads, _ in measured['calibration']])
                else:
                    cutoff = audit.calibration_cut(confidence['calibration'], suites['calibration'][0])
                outcomes = {}
                for suite, (rows, owners) in suites.items():
                    evidence = audit.single_evidence(field_extract, rows, head, owners, cutoff,
                                                     clause_confidence=confidence[suite])
                    outcome = audit.score_single(rows, evidence, field, intent, schema['cutoffs']['label'])
                    if suite in measured:
                        outcome.update(by_request_type(measured[suite], outcome, cutoff))
                    outcomes[suite] = outcome
                    print(name, cutoff, suite, outcome['intent_correct'], outcome['counts'], outcome.get('requests'), flush=True)
                results[name] = dict(cutoff=cutoff, outcomes=outcomes)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model input changed')
    store_json(args.output / 'result.json', dict(results=results, training=len(train), checkpoints=checkpoints,
                                               attention=args.attention, contextual=args.contextual,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('baseline', type=Path)
    cli.add_argument('focus', type=Path)
    cli.add_argument('schema', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--attention', choices=('recurrent', 'parent'), default='recurrent')
    cli.add_argument('--contextual', action='store_true')
    main(cli.parse_args())
