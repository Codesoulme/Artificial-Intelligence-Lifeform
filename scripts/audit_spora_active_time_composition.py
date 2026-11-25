"""Use the learned time selection and tensor-backed values in the same reply."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_active_time import ActiveTimeSelector, evaluate, features_for
from scripts.audit_spora_composition_training import get_versions, grade
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import reply
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, tokens
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_memory_in_ordinary_replies import BUDGET
from scripts.audit_spora_typed_composition import BINDING_PREFIX, SLOTS, arrangement
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_active_time_cases import Example, held_texts
from scripts.spora_memory_composition_cases import HELD
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_store import VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


@dataclass(frozen=True)
class Request:
    index: int
    example: Example
    retired: tuple[str, ...] = ()


def requests():
    for i, spec in enumerate(HELD):
        case = spec.scenario
        yield Request(i, Example('single', case.question, re.search(spec.time_pattern, case.question).group()))
        if spec.changed_question:
            span = re.search(spec.changed_pattern, spec.changed_question).group()
            yield Request(i, Example('single_changed', spec.changed_question, span))
    for i, family, text, when, retired in (
        (0, 'revision', '给赵茗初写条会面微信。群里最早说周一中午，已改到星期三傍晚，地点和要带的东西按已有登记，只给成品。',
         '星期三傍晚', ('周一中午',)),
        (1, 'quote', '给何青玟拟条提醒，把登记的地点和用品写上。旧稿里写“明早十点碰面”，这次敲定明早八点。',
         '明早八点', ('明早十点',)),
        (3, 'backup', '帮我发给顾绮言一句话，按今晚九点在约好的地方见，让他带登记的书。后天上午只是我还没提的备选。',
         '今晚九点', ('后天上午',)),
        (5, 'revision', '帮黎书然写一句取书通知。周五傍晚那一档已取消，新的时间是周二中午，饮品和书目都按先前登记。',
         '周二中午', ('周五傍晚',)),
        (0, 'ambiguous', '赵茗初说周三晚上或者星期五早上都可能，还等答复。地点和用品已经登记，先别把时间写进通知。',
         None, ('周三晚上', '星期五早上')),
        (3, 'canceled', '我取消了跟顾绮言约的今晚九点，重新见的时间没敲定。地点和书的安排先保留。',
         None, ('今晚九点',)),
        (5, 'missing', '给黎书然准备一句取书消息，按之前登记的书和饮品。取书时间我还没决定。',
         None, ()),
    ):
        yield Request(i, Example(family, text, when), retired)


def execute(model, tokenizer, head, codebook, store, selections, typed_reference):
    totals = Counter()
    catalog = store.catalog()
    records = {}
    for task, selection in zip(requests(), selections, strict=True):
        case = HELD[task.index].scenario
        if task.index not in records:
            records[task.index] = get_versions(store, catalog, case, tokenizer, codebook)
        versions = records[task.index]
        question = task.example.text
        time = selection['selected']
        expected = task.example.current
        wanted = [pair[1] for pair in case.values]
        stale = [pair[0] for pair in case.values]
        if any(value in question for value in (*wanted, *stale)):
            raise ValueError('A persistent value leaked into the user turn')
        if time is None:
            output = {'answer': '时间还没确定。你想约在什么时候？', 'copied': [], 'token_ids': []}
            passed = expected is None
        else:
            offset = selection['offsets']
            if question[offset[0]:offset[1]] != time:
                raise ValueError('Selected time is outside the user turn')
            time_tokens = tokens(tokenizer, time)
            if not time_tokens or tokenizer.decode(time_tokens) != time:
                raise ValueError('Selected time failed to round-trip through the tokenizer')
            facts = [(field, bank[1]) for field, bank in zip(case.fields, versions, strict=True)]
            candidates = (*facts, ('time', torch.tensor(time_tokens, device=model.device)))
            output = reply(model, tokenizer, head, question, arrangement(candidates),
                           require_coverage=True, cover_present_slots=True, max_new_tokens=BUDGET)
            checks = grade(output, wanted, stale, tokenizer.eos_token_id, re.escape(expected or '\0'),
                           HELD[task.index].action_groups, '|'.join(map(re.escape, task.retired)))
            passed = time == expected and checks['screened']
        if task.example.family in ('single', 'single_changed') and time == expected:
            arm = 'current' if task.example.family == 'single' else 'change_time'
            prior = typed_reference['rows'][task.index]['arms'][arm]['answers']['typed_time']
            if output['token_ids'] != prior['token_ids']:
                raise ValueError('A correctly located time did not reproduce the fixed-state result')
        totals.update(cases=1, time_correct=time == expected, passed=passed,
                      abstained=time is None, false_confirmation=expected is None and time is not None)
        yield {**selection, 'answer': output['answer'], 'token_ids': output['token_ids'],
               'screened': passed, 'chosen_fields': wanted, 'retired_times': task.retired}
    print(totals, flush=True)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if args.output.exists():
        raise FileExistsError(args.output)
    selector_artifact = args.selector / 'selector.pt'
    reader_artifact = args.typed / 'typed_time.pt'
    database = args.records / 'versions.sqlite3'
    paths = (selector_artifact, reader_artifact, database, args.selector / 'result.json', args.typed / 'result.json',
             args.records / 'result.json', Path(__file__), base.MODEL / 'tokenizer.json')
    sources = {str(path): digest(path) for path in paths}
    selector_result, typed_reference, prior = [json.loads(path.read_text(encoding='utf-8')) for path in paths[3:6]]
    if (sources[str(selector_artifact)] != selector_result['head_sha256']
            or sources[str(reader_artifact)] != typed_reference['head_sha256']['typed_time']
            or sources[str(database)] != prior['database_sha256']):
        raise ValueError('The learned components and version archive do not match their audits')
    writer_sha = next(value for path, value in prior['sources'].items() if Path(path).name == 'adapters.pt')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    selector = ActiveTimeSelector(model.config.hidden_size)
    selector.load_state_dict(torch.load(selector_artifact, map_location='cpu', weights_only=True))
    features = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    cases = [task.example for task in requests()]
    selections, _ = evaluate(selector.eval(), cases, *features_for(features, [case.text for case in cases]))
    saved = {row['text']: row['selected'] for row in selector_result['rows']}
    if any(row['selected'] != saved[row['text']] for row in selections if row['text'] in saved):
        raise ValueError('Reload changed selection on a previously observed prompt')
    state = torch.load(reader_artifact, map_location='cpu', weights_only=True)
    head = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=len(SLOTS))
    head.load_state_dict(state)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    binding = BINDING_PREFIX + writer_sha + ':' + base.WEIGHT_SHA
    with VersionStore(database, binding) as store:
        rows = list(execute(model, tokenizer, head.eval().to(model.device), codebook, store, selections, typed_reference))
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise ValueError('A frozen input changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store_json(args.output, {'sources': sources, 'rows': rows, 'seconds': perf_counter() - started,
                             'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selector', 'typed', 'records', 'output'):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
