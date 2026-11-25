"""Test whether version-local tensors can support a normal user-facing message."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dual_version_reply import FIELDS, reply
from scripts.audit_spora_fast_weight_lexical_write import SOURCE, base, compress_dimension
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_versioned_memory_generation import MEMORY_FRACTION, read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


BUDGET = 72
ARMS = {'current': (1, 1), 'swap_first': (0, 1), 'swap_second': (1, 0), 'both_old': (0, 0)}
DAYS = (date(2026, 10, 6), date(2026, 10, 14))


@dataclass(frozen=True)
class Case:
    name: str
    fields: tuple[str, str]
    values: tuple[tuple[str, str], tuple[str, str]]
    question: str


CASES = (
    Case('赵茗初', ('places', 'items'), (('白塔楼', '青瓦廊'), ('红雨伞', '铜书夹')),
         '给赵茗初写一句能直接发的微信，说明周五傍晚在哪里碰面、需要带什么。只给消息正文。'),
    Case('何青玟', ('items', 'places'), (('帆布包', '蓝围巾'), ('绿荫亭', '木栈桥')),
         '明早八点，何青玟要和我见面。根据已经登记的用品和见面地点，写一句发给她的提醒，请她带上那个用品。'),
    Case('许昀芝', ('drinks', 'books'), (('薄荷水', '红豆奶'), ('冬日海湾', '山谷晚风')),
         '给许昀芝发一句话，告诉她下午借的书已经准备好了，顺便说会给她带喜欢的饮品。把书名和饮品名写进消息。'),
    Case('顾绮言', ('books', 'places'), (('花园来信', '长夜灯塔'), ('松木台', '白杨馆')),
         '给顾绮言写一句短消息，请他今晚七点带已登记的那本书到约好的地方见。直接给能发送的一句话。'),
)


def writes(path, binding, model, tokenizer, columns, settings):
    with VersionStore(path, binding, create=True) as store:
        for case in CASES:
            for field, values in zip(case.fields, case.values, strict=True):
                bank = write_fresh(model, tokenizer, columns, layer=settings['layers'][0],
                                   eta=settings['eta'], ridge=settings['ridge'],
                                   events={field: ((case.name, *values),)},
                                   facts={field: '\n{value}'}, minimum_tokens=1)
                for index, ((keys, factors), when) in enumerate(zip(bank[case.name], DAYS, strict=True)):
                    store.append(f'{case.name}/{field}/{index}', MemoryAddress(case.name, field),
                                 when, when, LowRankFastWeight(keys, factors))


def output_metrics(output, expected, foils, eos):
    counts = tuple(output['answer'].count(value) for value in expected)
    stale = any(value in output['answer'] for value in foils)
    ended = bool(output['token_ids'] and output['token_ids'][-1] == eos)
    return {'values_once': tuple(count == 1 for count in counts), 'stale': stale, 'ended': ended,
            'both': all(count == 1 for count in counts) and not stale and ended}


def evaluate(store, model, tokenizer, head, codebook):
    records = store.catalog()
    if len(records) != len(CASES) * 2 or sum(map(len, records.values())) != len(CASES) * 4:
        raise ValueError('Unexpected number of fields or versions')
    rows, totals = [], {}
    for case in CASES:
        versions = []
        private = {value for pair in case.values for value in pair}
        if len(private) != 4 or any(value in case.question for value in private):
            raise ValueError('An evaluation value leaked into the fixed prompt')
        for field, expected in zip(case.fields, case.values, strict=True):
            entries = sorted(records[MemoryAddress(case.name, field)], key=lambda record: record.effective)
            if tuple(record.effective for record in entries) != DAYS:
                raise ValueError('Unexpected version dates in the reloaded bank')
            states = [(state.keys, state.values) for record in entries for state in (store.read(record.number),)]
            recovered = read_versions({field: states}, codebook)[field]
            if tuple(tokenizer.decode(value.tolist()) for value in recovered) != expected:
                raise ValueError('Stored tensors did not recover their expected values')
            versions.append(recovered)
        outputs = {}
        for arm, indices in ARMS.items():
            values, foils = [[case.values[slot][index ^ flip] for slot, index in enumerate(indices)]
                             for flip in (0, 1)]
            memory = tuple(data[index] for data, index in zip(versions, indices, strict=True))
            output = reply(model, tokenizer, head, case.question, memory, require_coverage=True,
                           max_new_tokens=BUDGET, memory_fraction=MEMORY_FRACTION)
            outputs[arm] = {**output, 'expected': values,
                            'metrics': output_metrics(output, values, foils, tokenizer.eos_token_id)}
        positive = [pair[1] for pair in case.values]
        negatives = [pair[0] for pair in case.values]
        labels = [FIELDS[field] for field in case.fields]
        prompt = case.question + '已确认的安排：' + '，'.join(
            f'{label}是{value}' for label, value in zip(labels, positive, strict=True)
        ) + '。'
        for mode, output in (
            ('weak_copy', reply(model, tokenizer, head, case.question, tuple(value[1] for value in versions),
                                require_coverage=True, max_new_tokens=BUDGET, memory_fraction=.5)),
            ('empty', reply(model, tokenizer, head, case.question, (None, None), max_new_tokens=BUDGET)),
            ('text_oracle', reply(model, tokenizer, head, prompt, (None, None), max_new_tokens=BUDGET)),
        ):
            outputs[mode] = {**output, 'expected': positive,
                             'metrics': output_metrics(output, positive, negatives, tokenizer.eos_token_id)}
        for mode, answer in outputs.items():
            metrics = answer['metrics']
            summary = totals.setdefault(mode, Counter())
            summary.update(first=metrics['values_once'][0], second=metrics['values_once'][1],
                           stale=metrics['stale'], ended=metrics['ended'], both=metrics['both'])
        rows.append({'name': case.name, 'question': case.question, 'oracle': prompt, 'outputs': outputs})
        print(case.name, {mode: output['metrics']['both'] for mode, output in outputs.items()}, flush=True)
    return rows, totals


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.replay:
        args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / ('replay.json' if args.replay else 'result.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    reference = args.reader / 'result.json'
    head_path, writer_path = args.reader / 'head.pt', SOURCE / 'adapters.pt'
    files = (reference, head_path, writer_path, base.MODEL / 'model.safetensors',
             base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json', Path(__file__))
    sources = {str(path): digest(path) for path in files}
    prior = json.loads(reference.read_text(encoding='utf-8'))
    if (any(sources.get(path) != expected for path, expected in prior['sources'].items())
            or sources[str(head_path)] != prior['head_sha256']):
        raise ValueError('Readout or writer files do not match the reference audit')
    binding = 'ordinary-reply-audit:' + sources[str(head_path)] + ':' + sources[str(writer_path)]
    path = args.output / 'versions.sqlite3'
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    payload = torch.load(head_path, map_location='cpu', weights_only=True)
    head = VersionReplyActions(model.config.hidden_size, payload['scaffold_tokens'], slots=2)
    head.load_state_dict(payload)
    head.eval().to(model.device)
    if not args.replay:
        settings = torch.load(writer_path, map_location='cpu', weights_only=True)
        writes(path, binding, model, tokenizer, columns, settings)
    before = digest(path)
    with VersionStore(path, binding) as store:
        rows, counts = evaluate(store, model, tokenizer, head, codebook)
    if digest(path) != before or any(digest(Path(file)) != expected for file, expected in sources.items()):
        raise ValueError('A protected input or the version store changed during recall')
    result = {'rows': rows, 'counts': counts, 'sources': sources, 'database_sha256': before,
              'budget': BUDGET, 'peak_bytes': torch.cuda.max_memory_allocated(), 'seconds': perf_counter() - started}
    if args.replay:
        previous = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(json.loads(json.dumps(result[key])) != previous[key]
               for key in ('sources', 'database_sha256', 'budget', 'rows', 'counts')):
            raise ValueError('Independent reloading changed the recalled outputs or their evidence')
    store_json(result_path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--replay', action='store_true')
    main(parser.parse_args())
