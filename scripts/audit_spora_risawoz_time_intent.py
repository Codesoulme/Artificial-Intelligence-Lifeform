"""Measure schedule false triggers on natural RiSAWOZ information requests."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from scripts.audit_spora_active_time import ActiveTimeSelector, candidates
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_risawoz_updates import DATA_MEMBER, SOURCE_HASHES
from scripts.audit_spora_time_authority import IntentAndRole, classify_all, views
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_active_time_cases import Example
from scripts.train_spora_recurrent_counterfactual import store_json


# Reviewed development references. Conversation text stays in the source archive.
CASES = (
    ('relative_query', 'train_goal_2-24###1649', 4),
    ('relative_query', 'train_goal_2-33_v2###10652', 4),
    ('relative_query', 'train_goal_1-32###1051', 0),
    ('query_followup', 'train_goal_1-42_v2###201', 1),
    ('relative_query', 'train_goal_4-64_v2###10621', 0),
    ('query_change', 'train_goal_2-18_v2###11073', 3),
    ('conditional_query', 'train_goal_2-30_v2###10757', 4),
    ('conditional_query', 'flight_goal_2-5###1131', 3),
    ('relative_query', 'flight_goal_3-22###1049', 0),
    ('conditional_query', 'flight_goal_2-40###1737', 3),
    ('conditional_query', 'flight_goal_2-18###481', 6),
    ('query_followup', 'flight_weather_goal_3-32###762', 1),
    ('conditional_query', 'flight_weather_goal_3-32###762', 7),
    ('query_change', 'train_weather_goal_2-55_v2###10961', 6),
    ('weather', 'weather_goal_1-4###1748', 0),
    ('weather', 'weather_goal_3-3###1048', 3),
    ('weather', 'weather_goal_2-26###7012', 0),
    ('weather_followup', 'weather_goal_3-14_v2###3278', 3),
    ('weather_followup', 'weather_goal_3-44_v2###3157', 3),
    ('weather_followup', 'weather_goal_2-25###1815', 2),
    ('weather_change', 'weather_goal_3-42_v2###10925', 3),
    ('options', 'Class_goal_5-41_v2###9020', 0),
    ('options', 'Class_goal_5-36###4489', 0),
    ('options', 'Class_goal_3-46###3691', 2),
    ('non_schedule', 'hotel_goal_1-59###5076', 0),
    ('non_schedule', 'movie_goal_1-64###6857', 0),
)


def examples(archive, split_file):
    if (digest(archive), digest(split_file)) != (SOURCE_HASHES[0], SOURCE_HASHES[2]):
        raise ValueError('Expected the pinned official RiSAWOZ corpus')
    with ZipFile(archive) as source, source.open(DATA_MEMBER) as stream:
        dialogues = json.load(stream)
    corpus = {row['dialogue_id']: row['dialogue'] for row in dialogues}
    splits = json.loads(split_file.read_text(encoding='utf-8'))
    if (len(corpus) != len(dialogues) or set(corpus) != set(splits['all_dev600.json'])
            or set(corpus) & (set(splits['all_train10000.json']) | set(splits['all_test600_new.json']))):
        raise ValueError('Development dialogue split does not match its manifest')
    observed = []
    for group, dialogue, turn_id in CASES:
        turn, = [row for row in corpus[dialogue] if row['turn_id'] == turn_id]
        text = turn['user_utterance']
        if not candidates(text):
            raise ValueError('A selected audit request lost all time candidates')
        observed.append(Example(group, text, None))
    return observed


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if args.output.exists():
        raise FileExistsError(args.output)
    archive, split = args.corpus / 'source-data.zip', args.corpus / 'dataset_split_ids_dict.json'
    cases = examples(archive, split)
    registry = {'old_selector': args.old / 'selector.pt', 'paired_selector': args.study / 'paired_selector.pt',
                'intent_and_role': args.study / 'intent_and_role.pt'}
    report = args.study / 'result.json'
    prior = json.loads(report.read_text(encoding='utf-8'))
    files = (archive, split, report, *registry.values(), Path(__file__), base.MODEL / 'tokenizer.json')
    sources = {str(path): digest(path) for path in files}
    if any(sources[str(path)] != prior['heads'][mode] for mode, path in registry.items()):
        raise ValueError('The time selector weights do not match their independent audit')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    heads = {}
    for mode, path in registry.items():
        head = (IntentAndRole if mode == 'intent_and_role' else ActiveTimeSelector)(model.config.hidden_size)
        head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        heads[mode] = head.eval()
    rows = classify_all(heads, cases, *views(extract, cases))
    summary = {}
    for mode, group in rows.items():
        summary[mode] = sum(item['selected'] is not None for item in group)
        for row, (_, dialogue, turn) in zip(group, CASES, strict=True):
            row.update(dialogue=dialogue, turn=turn)
        print(mode, summary[mode], '/', len(group), flush=True)
    if any(digest(path) != sources[str(path)] for path in files):
        raise ValueError('A protected source changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store_json(args.output, {'sources': sources, 'rows': rows, 'false_schedule_confirmations': summary,
                             'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'old', 'study', 'output'):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
