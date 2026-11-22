"""Seal calendar history and verify native DeepSeek reads before and after waking."""

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import WORDS
from scripts.audit_spora_calendar_growth import check
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.durable import TensorStore
from spora.growing import scalars
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest
from spora.sleep_memory import SealedPage


def main(args):
    started = time.perf_counter()
    torch.set_num_threads(2)
    destination = args.output / ('reopened.json' if args.reopen else 'result.json')
    database = args.output / 'individual.sqlite'
    if destination.exists() or (not args.reopen and args.output.exists()):
        raise FileExistsError(destination)
    source_sha = sha(args.checkpoint)
    prior_path = args.source / 'result.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    source_db = args.source / 'individual.sqlite'
    source_database_sha = sha(source_db)
    if not args.reopen:
        args.output.mkdir(parents=True)
        with TensorStore(source_db) as seed:
            seed.backup(database)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with load_integrated_durable(database, args.checkpoint, load_full_streamed().eval(), MODEL) as runtime:
        if args.reopen:
            first = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
            wake = first['wake']
        else:
            if frozen_digest(runtime.organism, runtime.state) != prior['state_sha256']:
                raise ValueError('The sleep source differs from the verified growth run')
            baseline = check(runtime, tokenizer, prior['boundaries'])
            if baseline != prior['reads'] or not all(row['passed'] for row in baseline):
                raise ValueError('The decoder baseline did not reproduce the expected historical answers')
            runtime.store.compact()
            before_bytes, before_scalars = database.stat().st_size, scalars(runtime.state)
            original, revision = runtime.state, runtime.store.revision
            rejected = runtime.consolidate_history(0, rank=16, steps=96)
            if rejected['accepted'] or runtime.state is not original or runtime.store.revision != revision:
                raise ValueError('The strict gate did not preserve the parent after the failed trained proposal')

            def verify(candidate):
                return check(runtime, tokenizer, prior['boundaries'], state=candidate) == baseline

            seals = []
            for index in (0, 1):
                result = runtime.consolidate_history(index, rank=8, method='svd', verify=verify)
                if not result['accepted']:
                    raise ValueError(f'Page {index} failed sleep verification: {result}')
                seals.append(result)
                print(f'Sealed history page {index}', flush=True)
            runtime.store.compact()
            after_bytes, after_scalars = database.stat().st_size, scalars(runtime.state)
            day = runtime.observation_calendar.days[-1] + timedelta(days=1)
            key = runtime.organism.identity_fingerprints.lookup('小林')
            event = runtime.state.cursor
            decision, stats = runtime.observe_sourced(key, WORDS.index('酸奶'), 'direct', 1., observed_on=day)
            if not decision['current_write'] or decision['history_time'] != event:
                raise ValueError('The first observation after sleep was not learned')
            wake = {'event': event, 'day': day.isoformat(), 'value': '酸奶', 'write': stats}

        digest = frozen_digest(runtime.organism, runtime.state)
        rows = check(runtime, tokenizer, [*prior['boundaries'], wake])
        if not all(row['passed'] for row in rows) or digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('History or the new observation changed after sleep')
        phases = [isinstance(page, SealedPage) for page in runtime.state.pages]
        if phases != [True, True, False]:
            raise ValueError('Expected two sealed history pages and one active page')
        report = {'wake': wake, 'reads': rows, 'sealed': phases, 'state_sha256': digest,
                  'revision': runtime.store.revision, 'observations': runtime.state.cursor,
                  'source_sha256': source_sha, 'source_database_sha256': source_database_sha}
        if args.reopen:
            if any(first[key] != value for key, value in report.items()):
                raise ValueError('Independent load did not reproduce the consolidation result')
            report['exact_reopen'] = True
        else:
            report.update(compacted_bytes_before=before_bytes, compacted_bytes_after=after_bytes,
                          scalars_before=before_scalars, scalars_after=after_scalars,
                          rejected_training=rejected, sleep=seals)
    if sha(args.checkpoint) != source_sha or sha(source_db) != source_database_sha:
        raise ValueError('The audit changed source artifacts')
    report['seconds'] = time.perf_counter() - started
    write(destination, report)
    print(f'Verified {len(rows)} reads at {report["observations"]} observations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('source', 'checkpoint', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    main(parser.parse_args())
