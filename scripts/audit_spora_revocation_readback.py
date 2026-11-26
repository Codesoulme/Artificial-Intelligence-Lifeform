"""Measure whether negation and speaker examples improve real version readback."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_confirmed_revocation import (
    REVOKE_BLIND, REVOKE_HELD, REVOKE_TRAIN, VERSION_BLIND, VERSION_TRAIN, classify, fit_probe,
    revocation_examples, rows_from, score,
)
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_lookup_transitions import normalize_row
from scripts.audit_spora_recurrent_lookup_scope import ASK, CHAT, READ, describe, episodes, evaluate
from scripts.spora_version_chat import FIELDS, VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


HARD_TRAIN = (
    ('不许删去旧任务，还是查它', '旧任务现在从查询中删去'),
    ('免得漏掉，请不要撤回早先那件', '请把早先那件的查阅撤回'),
    ('别取消上一项，按原定执行', '取消上一项，从本次安排划除'),
    ('我的意思是不要把第一件移出范围', '我的意思是把第一件移出范围'),
    ('朋友说停掉原来的，可我要继续查', '朋友说继续查，可我决定停掉原来的'),
    ('有人建议撤销，实际我仍想核对前项', '有人建议继续，实际我决定撤销前项'),
    ('「旧单作废」是别人的发言，我的原单照旧', '「旧单照旧」是别人的发言，我的原单作废'),
    ('他们不查先提的那笔，我这边坚持查', '他们还查先提的那笔，我这边决定不查'),
)
HARD_CAL = (
    ('不要划掉当初选的那份', '当初选的那份可以从本轮划掉'),
    ('同事叫我跳过旧项，我照查', '同事叫我照查旧项，我决定跳过'),
    ('原定的请求这回也执行', '原定的请求这回不用执行'),
    ('别因为那句取消就把头一件踢出去', '头一件直接从这次核对中踢出去'),
)
TRANSFER = (
    ('不要把开头那份从待查单挪掉', '把开头那份从待查单挪出去'),
    ('第一份得留着，别给去掉', '第一份这回得去掉'),
    ('别听他讲的「前一笔不查」，我还要核实', '别听他讲的「前一笔照查」，我这次决定撤销'),
    ('她建议把前头那项勾掉，但我还是要看', '她建议把前头那项留着，但我已经不打算看'),
    ('最先提出的那件要一起办', '最先提出的那件到这里就结了，不用办'),
    ('「把原项目撤下」是我举的例子，实际要留下', '「把原项目留下」是我举的例子，实际要撤下'),
)


def training_rows(families):
    return [(f'{prefix}，{phrase}。', label) for prefix in ('查本项记录', '也查另一项记录', '请接着核对')
            for phrase, label in revocation_examples(families)]


def thresholds(rows):
    def objective(cutoff):
        accepted = [row for row in rows if row['predicted'] == 1 and row['confidence'] >= cutoff]
        good = sum(row['expected'] == 1 for row in accepted)
        bad = len(accepted) - good
        return good - 3 * bad, -bad, -cutoff

    return max((.5, .65, .8, .9, .95, .99, 1.01), key=objective)


def readback_rows():
    seeds = [row for row in episodes('held') if row.scope == 'replace' and row.question == '首版。'][::2]
    cases = []
    for index, pair in enumerate(TRANSFER):
        seed = seeds[index % len(seeds)]
        version = index % 3
        question = VERSION_BLIND[version][index // 3]
        for revoked, phrase in enumerate(pair):
            request = f'请查{describe(seed.addresses[1])}，{phrase}。'
            row = replace(seed, scope='replace' if revoked else 'both', events=(seed.events[0], request),
                          question=question, action=READ + 1 if revoked else ASK, version=version if revoked else -1)
            cases.append(row)
    return cases


@torch.inference_mode()
def replay(updates, versions, cut, cases, store, model, tokenizer, codebook, reader, values):
    plans = []
    for update, version_result in zip(updates, versions, strict=True):
        read = update['predicted'] == 1 and update['confidence'] >= cut
        target = version_result['predicted']
        action = CHAT if target == 3 else READ + 1 if read else ASK
        plans.append((action, target))
    result = evaluate(cases, plans, store.catalog(), store, model, tokenizer, codebook, reader, values=values)
    correct_pairs = 0
    for case, keep, switch in zip(cases[1::2], result['rows'][::2], result['rows'][1::2], strict=True):
        record = switch['expected_record']
        label = FIELDS[case.addresses[1].field]
        correct_pairs += (keep['decision'][0] == ASK and keep['selected'] is None
                          and switch['selected'] == record and switch['decision'][1] == case.version
                          and switch['answer'] == f'{label}：{values[record]}。')
    result.update(revocation=updates, versions=versions, correct_pairs=correct_pairs)
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    archive = args.archive / 'three-versions.sqlite3'
    reader_path, normalizer, writer, binding = artifacts(args.reader, args.normalizer)
    protected = (archive, reader_path, normalizer, writer, args.recurrent[0],
                 *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in protected}
    if sources[archive] != prior['archive_sha256']:
        raise ValueError('Three-version archive does not match its report')
    prior_sources = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    if any(prior_sources.get(path.resolve()) != sources[path] for path in protected[1:]):
        raise ValueError('Model or reader differs from the archive experiment')
    values = {int(number): value for number, value in prior['values'].items()}
    cases = readback_rows()
    if any(value in text for row in cases for text in (*row.events, row.question) for value in values.values()):
        raise ValueError('An evaluation request contains an archived value')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        version = fit_probe(extract, rows_from(VERSION_TRAIN), 4)
        atomic_save(version.state_dict(), args.output / 'version.pt')
        weights = torch.load(reader_path, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        columns = compress_dimension(embedding.shape[1]).to(embedding.device)
        codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
        for name, families in (('base', REVOKE_TRAIN), ('augmented', REVOKE_TRAIN + HARD_TRAIN)):
            head = fit_probe(extract, training_rows(families), 2, clauses=True)
            checkpoint = args.output / f'{name}.pt'
            atomic_save(head.state_dict(), checkpoint)
            head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            calibration = classify(head, extract, revocation_examples(HARD_CAL), clauses=True)
            cut = thresholds(calibration)
            wrapped_calibration = classify(head, extract, [(f'请查另一项记录，{text}。', label)
                                            for text, label in revocation_examples(HARD_CAL)], clauses=True)
            wrapped_cut = thresholds(wrapped_calibration)
            old_tests = classify(head, extract, revocation_examples(REVOKE_HELD + REVOKE_BLIND), clauses=True)
            new_tests = classify(head, extract, revocation_examples(TRANSFER), clauses=True)
            updates = classify(head, extract, [(normalize_row(row)[0].events[1], int(row.scope == 'replace'))
                                               for row in cases], clauses=True)
            versions = classify(version, extract, [(row.question, row.version) for row in cases])
            readbacks = {}
            for mode, threshold in (('standalone', cut), ('wrapped', wrapped_cut), ('uncalibrated', .5)):
                readbacks[mode] = replay(updates, versions, threshold, cases, store, model, tokenizer, codebook, reader, values)
                print(name, mode, threshold, readbacks[mode]['summary'], flush=True)
            results[name] = dict(calibration=calibration, threshold=cut, wrapped_calibration=wrapped_calibration,
                                 wrapped_threshold=wrapped_cut, old=score(old_tests), new=score(new_tests),
                                 readbacks=readbacks, new_rows=new_tests, checkpoint_sha256=base.sha(checkpoint))
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model or archive changed')
    store_json(args.output / 'result.json', dict(results=results, sources={str(path): hash for path, hash in sources.items()},
                                               seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    main(cli.parse_args())
