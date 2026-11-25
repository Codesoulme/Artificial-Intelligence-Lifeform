"""Compare frozen spoken planners on matched requests with and without a known coordinate."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_read_confirmation_action as action
from scripts import audit_spora_spoken_coordinate_clarification as coordinate
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_spoken_coordinate_clarification import ADDRESS, ASK_FIELD, ASK_OWNER, CHAT
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_dialogue_scope import PartialLookup
from spora.research_version_store import MemoryAddress


HELD = (
    (ASK_OWNER, 0, 'drinks', '那杯已经说定的是什么来着？姓名我再告诉你。'),
    (ADDRESS, 0, 'drinks', '夏宁瑶那杯已经说定的是什么来着？我出门前得知道。'),
    (ASK_OWNER, 1, 'places', '说好的会合处在哪边，我得弄清。具体是谁先让我想想。'),
    (ADDRESS, 1, 'places', '我和夏宁遥说好的会合处在哪边，我得弄清。'),
    (ASK_OWNER, 2, 'items', '原来交代我拿去的物件叫什么？我等下告诉你问谁。'),
    (ADDRESS, 2, 'items', '郭回霜原来交代我拿去的物件叫什么？我得照她说的带过去。'),
    (ASK_OWNER, 3, 'books', '先确认旧单上选了哪本，我一会儿说明是谁挑的。'),
    (ADDRESS, 3, 'books', '先确认郭回双旧单上选了哪本，我准备去取那一册。'),
    (ASK_FIELD, 0, 'books', '夏宁瑶当时留的答案我得看一眼，具体哪方面我还没说。'),
    (ASK_FIELD, 1, 'items', '麻烦回头找找夏宁遥先前的那笔，是什么类别我稍后补。'),
    (ASK_FIELD, 2, 'places', '我忘了郭回霜先前记过的一项。我先把人说了，哪项请等我补。'),
    (ASK_FIELD, 3, 'drinks', '郭回双原话我想找回来。只记得是谁说的，还没给你要找的条目。'),
    (ADDRESS, 1, 'books', '夏宁瑶那本先放着。夏宁遥当时选中的是哪一册？'),
    (ADDRESS, 0, 'places', '郭回双另有事，跟夏宁瑶一早说好的集合位置是哪儿？'),
    (ADDRESS, 3, 'items', '别管郭回霜那一件，郭回双让我准备的物件倒是叫什么呢？'),
    (ASK_OWNER, 2, 'drinks', '夏宁遥那杯我已经买了。另一杯还不知道选哪种，具体是谁的我再说。'),
    (ASK_OWNER, 0, 'books', '郭回双我刚见过。另一个人以前挑的是哪册？姓名我等会给你。'),
    (CHAT, None, None, '那杯已经说定的我记得。姓名我等会儿填，你先帮我想一句问候。'),
    (CHAT, None, None, '夏宁遥挑的那本先不用翻，我自己去问。'),
    (CHAT, None, None, '郭回霜说过的东西我手里有。你帮我解释一下怎么打包。'),
    (CHAT, None, None, '我想礼貌地问郭回双选哪一册，你帮我把问题写出来。'),
    (CHAT, None, None, '请把夏宁瑶之前约在哪儿这句话改成更客气的语气。'),
    (CHAT, None, None, '碰面地儿别回头找了，我打算再约一个。'),
    (CHAT, None, None, '姓名和细目我稍后给客服。先告诉我这张表该怎么填。'),
)


def reconcile(partial, full, mentions):
    """Treat a matching whole-address score and unique mention as joint evidence."""
    if (isinstance(partial, PartialLookup) and partial.owner is None and len(mentions) == 1
            and full == MemoryAddress(mentions[0], partial.field)):
        return full
    return partial or full


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    baseline_report = json.loads((args.whole / 'result.json').read_text(encoding='utf-8'))
    coordinate_report = json.loads((args.coordinate / 'result.json').read_text(encoding='utf-8'))
    archive = args.profile / 'profile' / 'versions.sqlite3'
    reader_file, _, writer_file, binding = action.artifacts(args.reader, args.normalizer)
    pointer_file = args.transfer / 'personal.pt'
    checkpoint, *layers = args.recurrent
    files = (archive, reader_file, writer_file, pointer_file, checkpoint, args.whole / 'action-19.pt',
             *(folder / 'attention.pt' for folder in layers))
    known = {Path(path).resolve(): checksum for path, checksum in coordinate_report['sources'].items()}
    known[(args.coordinate / 'action.pt').resolve()] = coordinate_report['action_sha256']
    paths = (*files, args.coordinate / 'action.pt')
    if any(known.get(path.resolve()) != action.digest(path) for path in paths):
        raise ValueError('Expected the original frozen heads and archive')
    args.output.mkdir(parents=True, exist_ok=False)
    questions = [question for _, _, _, question in HELD]
    held = {plan.anonymize(question, dense.NAMES)[0] for question in questions}
    previous = {plan.anonymize(question, dense.NAMES)[0] for _, _, _, question in coordinate.HELD}
    for frames, names, aliases in ((coordinate.TRAIN, action.TRAIN_NAMES, dense.ALIASES),
                                   (coordinate.CAL, action.CAL_NAMES, dense.CAL_ALIASES)):
        previous.update(plan.anonymize(question, names)[0]
                        for question, _ in coordinate.supervise(frames, names, aliases))
    if held & previous or len(held) != len(questions):
        raise ValueError('Repeated prompts in the fresh replay')

    tokenizer = action.base.AutoTokenizer.from_pretrained(action.base.MODEL, local_files_only=True, trust_remote_code=False)
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    with action.local_model(args.recurrent) as model, action.VersionStore(archive, binding) as store:
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = action.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        pointer = dense.JointPointer(model.config.hidden_size).eval()
        pointer.load_state_dict(torch.load(pointer_file, map_location='cpu', weights_only=True)['pointer'])
        early = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        deep = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=19)
        feature_rows = dense.align(early, questions, dense.NAMES)
        addresses, _ = action.rank(pointer, feature_rows, store.catalog())
        states = action.action_features(deep, questions, dense.NAMES)
        full_head = nn.Sequential(nn.LayerNorm(states.shape[1]), nn.Linear(states.shape[1], 64),
                                  nn.GELU(), nn.Linear(64, 1))
        full_head.load_state_dict(torch.load(args.whole / 'action-19.pt', weights_only=True, map_location='cpu'))
        full_head.eval()
        partial_head = coordinate.new_head(states.shape[1])
        partial_head.load_state_dict(torch.load(args.coordinate / 'action.pt', weights_only=True, map_location='cpu'))
        partial_head.eval()
        full_scores = action.predict(full_head, states)
        partial_scores = coordinate.scores(partial_head, states)
        whole, targeted, union, grounded, decisions = {}, {}, {}, {}, []
        for question, direct, mentions, confidence, (label, score) in zip(
                questions, addresses, feature_rows[2], full_scores, partial_scores, strict=True):
            complete = direct if confidence >= baseline_report['results']['19']['cutoff'] else None
            partial = (coordinate.proposed(label, mentions, direct)
                       if score >= coordinate_report['threshold'] else None)
            combined = reconcile(partial, complete, mentions)
            whole[question], targeted[question] = complete, partial
            union[question], grounded[question] = partial or complete, combined
            decisions.append(dict(question=question, full_score=confidence, predicted_action=label, action_score=score,
                                  mentions=mentions,
                                  choices=[asdict(item) if item is not None else None
                                           for item in (complete, partial, combined)]))
        results = {}
        for name, proposals in (('whole', whole), ('targeted', targeted), ('union', union), ('grounded', grounded)):
            data = coordinate.replay(store, model, tokenizer, reader, settings, proposals, cases=HELD)
            results[name] = data
            print(name, data['summary'], flush=True)
    if any(known[path.resolve()] != action.digest(path) for path in paths):
        raise RuntimeError('A source head or archive changed during replay')
    store_json(args.output / 'result.json', dict(results=results, decisions=decisions, seconds=perf_counter() - started,
               sources={str(path): known[path.resolve()] for path in paths}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('whole', type=Path)
    parser.add_argument('coordinate', type=Path)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
