"""Prepare unchanged, whole TRAIN-shard stories; never approve or train them.

Writes exclusively a fresh runs/p28-stories-data/<name>/ and the designated
report. The inherited tokenizer/held/dialogue/focus payloads are byte copies.
New story held splits are separate from the inherited development/test files.
"""
from __future__ import annotations
import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
# Also bounds Windows process affinity to at most two CPUs. No model imports.
from scripts.prepare_p28_prose import (
    HeldIndex, load_old_held, normalized, signatures, digest, file_hash,
)
os.environ['ARROW_NUM_THREADS'] = '1'
BASE = ROOT/'runs/p28-stories-data'
REPORT = ROOT/'reports/P28-stories-preparation.json'
MATERIAL = ROOT/'runs/p28-stories-material'
OLD = ROOT/'runs/conversation-data-v5'
CURRICULUM = ROOT/'runs/teaching-curriculum-v3'
VENDOR = ROOT/'runs/p28-source-review/vendor'
REPOSITORY = 'Gabrui/multilingual_TinyStories'
REVISION = '7fb5f327b907318dd95877a366e288f1971b64c1'
SOURCE_PATH = 'chinese/train-00000-of-00005.parquet'
SOURCE_SHA = '3c86b047b0063056f11b17a494c20a79ce8fda2f17bf4d06634a9d332b740302'
SEED = 281109
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
LATIN = re.compile(r'[A-Za-z]')
FOREIGN = re.compile(r'[\u0400-\u052f\u0600-\u06ff\u0900-\u097f\u3040-\u30ff\uac00-\ud7af]')
ENDING = re.compile(r'[。！？.!?][”’」』）)\]】"\']*$')


def output_dir(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use a simple fresh output name')
    return BASE/name


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def arrow_modules():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import pyarrow as pa
    import pyarrow.parquet as pq
    if pa.__version__ != '19.0.1' or VENDOR.resolve() not in Path(pa.__file__).resolve().parents:
        raise ValueError('Require the isolated existing pyarrow19.0.1 vendor')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return pa, pq


def verified_source():
    path = MATERIAL/'provenance.json'
    if not path.is_file():
        raise FileNotFoundError('Ask main for completed runs/p28-stories-material/provenance.json; do not use partial download')
    provenance = json.loads(path.read_text('utf-8'))
    required = {'repository': REPOSITORY, 'revision': REVISION, 'path': SOURCE_PATH,
                'split': 'train', 'sha256': SOURCE_SHA, 'local_file': Path(SOURCE_PATH).name}
    for key, value in required.items():
        if provenance.get(key) != value:
            raise ValueError('Pinned source provenance mismatch: '+key)
    source = MATERIAL/provenance['local_file']
    if source.stat().st_size != provenance['bytes'] or file_hash(source) != SOURCE_SHA:
        raise ValueError('Downloaded TRAIN shard integrity mismatch')
    if not provenance.get('declared_license'):
        raise ValueError('Missing declared source license')
    return source, provenance


def iter_stories(parquet):
    """Bounded batches, stable zero-based physical row ordinal; one column only."""
    ordinal = 0
    for batch in parquet.iter_batches(batch_size=512, columns=['story'], use_threads=False):
        for text in batch.column(0).to_pylist():
            yield ordinal, text
            ordinal += 1


def quality_reasons(text):
    if not isinstance(text, str):
        return ['not_text']
    stripped = text.strip()  # inspection only; output and encoding keep original.
    if len(stripped) < 60:
        return ['under_60_chars']
    if len(text) > 10000:
        return ['unreasonable_character_length']
    reasons = []
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\ud800-\udfff]', text):
        reasons.append('malformed_character')
    if re.search(r'<\||<[^>]+>|https?://|```|\{\{|\[\[', text):
        reasons.append('markup_or_control_text')
    if not ENDING.search(stripped) or stripped.endswith(('...', '……')):
        reasons.append('unfinished_ending')
    compact = re.sub(r'\s+', '', text)
    han, latin = len(HAN.findall(text)), len(LATIN.findall(text))
    if han/max(1, len(compact)) < .60:
        reasons.append('low_chinese_ratio')
    if latin/max(1, han+latin) > .08 or len(FOREIGN.findall(text))/max(1, len(compact)) > .02:
        reasons.append('mixed_language')
    if re.search(r'(.)\1{5,}|(.{2,24})\2{3,}', compact):
        reasons.append('repeated_run')
    sentences = [normalized(s) for s in re.split(r'[。！？!?]', text) if len(s.strip()) >= 12]
    if len(sentences) != len(set(sentences)):
        reasons.append('repeated_sentence')
    grams = [compact[i:i+8] for i in range(max(0, len(compact)-7))]
    if grams and 1-len(set(grams))/len(grams) > .25:
        reasons.append('repeated_8gram')
    for left, right in (('（', '）'), ('(', ')'), ('“', '”'), ('「', '」'), ('『', '』'), ('《', '》')):
        balance = 0
        for char in text:
            balance += (char == left)-(char == right)
            if balance < 0:
                break
        if balance != 0:
            reasons.append('unbalanced_delimiter')
            break
    return sorted(set(reasons))


def identities(text):
    key = normalized(text)
    return digest(key), digest(key[:96])


def split_of_group(group):
    bucket = int(group[:16], 16) % 1000
    return 'test' if bucket < 10 else 'development' if bucket < 20 else 'train'


def encode_whole(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or len(ids) > 510:
        return None, 'over_510_content_tokens'
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        return None, 'tokenizer_roundtrip'
    return ids, None


def create_selection_db(db):
    db.execute('PRAGMA cache_size=-8192')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE candidates (full_hash TEXT PRIMARY KEY, group_hash TEXT, score TEXT, split TEXT, row_index INTEGER UNIQUE, original_sha TEXT, content_tokens INTEGER, held_text TEXT)')


def add_candidate(db, text, row_index, content_tokens):
    full, group = identities(text)
    split = split_of_group(group)
    try:
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)',
            (full, group, digest(f'{SEED}:{full}'), split, row_index, digest(text),
             content_tokens, text if split != 'train' else None))
    except sqlite3.IntegrityError:
        return False
    return True


def ranked(db, split, limit):
    rows = db.execute('SELECT full_hash,group_hash,row_index,original_sha,content_tokens,held_text FROM candidates WHERE split=? ORDER BY score LIMIT ?', (split, limit))
    return [{'sha256': r[0], 'group': r[1], 'source_row_index': r[2], 'source_original_sha256': r[3],
             'content_tokens': r[4], 'text': r[5], 'split': split} for r in rows]


class MatchingHeld:
    """Same finite legacy rule, with entry numbers for removing held conflicts."""
    def __init__(self, records):
        self.index = HeldIndex()
        self.exact = defaultdict(set)
        self.groups = defaultdict(set)
        for i, row in enumerate(records):
            full, group = identities(row['text'])
            self.exact[full].add(i)
            self.groups[group].add(i)
            self.index.add(row['text'])

    def matches(self, text):
        full, group = identities(text)
        result = set(self.exact.get(full, ())) | set(self.groups.get(group, ()))
        sigs = signatures(text)
        shared = Counter(i for sig in sigs for i in self.index.index.get(sig, ()))
        result.update(i for i, n in shared.items() if min(len(sigs), self.index.sizes[i]) >= 8
                      and n/min(len(sigs), self.index.sizes[i]) >= .8)
        return result


def remove_training_conflicts(records, tokenizer, stats):
    """Exclude new held against actual dialogue/focus/teacher/foundation inputs.

    No claim about the historical pretraining stream or every ancestor corpus.
    """
    matcher = MatchingHeld(records)
    bad = set()
    for tag, path in (('inherited_dialogue', OLD/'dialogue-train.jsonl'),
                      ('teacher', CURRICULUM/'teacher.jsonl'),
                      ('foundations', CURRICULUM/'foundations.jsonl')):
        prior = len(bad)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                bad.update(matcher.matches(row['text']))
                for message in row.get('messages', ()):
                    bad.update(matcher.matches(message['content']))
                stats[tag+'_rows_scanned_for_new_held'] += 1
        stats['new_held_removed_'+tag+'_overlap'] = len(bad)-prior
    for seq in json.loads((OLD/'focus-tokens.json').read_text('utf-8')):
        bad.update(matcher.matches(tokenizer.decode(seq, skip_special_tokens=True)))
    stats['new_held_removed_current_training_overlap'] = len(bad)
    return [row for i, row in enumerate(records) if i not in bad]


def separate_new_held(records, stats):
    test = [r for r in records if r['split'] == 'test']
    test_index = HeldIndex()
    for row in test:
        test_index.add(row['text'])
    kept = list(test)
    for row in records:
        if row['split'] != 'development':
            continue
        why = test_index.reason(row['text'])
        if why:
            stats['development_removed_new_test_'+why] += 1
        else:
            kept.append(row)
    new_held = HeldIndex()
    for row in kept:
        new_held.add(row['text'])
    return kept, new_held


def write_story(raw, binary, row, ids, bos, eos, offset):
    seq = [bos]+ids+[eos]
    packed = array('H', seq)
    if packed.itemsize != 2:
        raise ValueError('uint16 platform required')
    if sys.byteorder != 'little':
        packed.byteswap()
    row = dict(row, token_offset=offset, token_count=len(seq))
    binary.write(packed.tobytes())
    raw.write(json.dumps(row, ensure_ascii=False)+'\n')
    return row


def copy_inherited(out, manifest):
    copied = {}
    for name, expected in manifest['files'].items():
        if name in ('pretrain-train.bin', 'pretrain-train.jsonl'):
            continue
        if Path(name).name != name:
            raise ValueError('Unexpected inherited artifact path')
        source = OLD/name
        if file_hash(source) != expected:
            raise ValueError('Inherited artifact hash mismatch: '+name)
        destination = out/name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        if file_hash(destination) != expected:
            raise ValueError('Byte copy differs: '+name)
        copied[name] = expected
    for name in ('manifest.json', 'lineage-audit.json'):
        source, destination = OLD/name, out/('inherited-v5-'+name)
        shutil.copyfile(source, destination)
        if file_hash(source) != file_hash(destination):
            raise ValueError('Inherited metadata copy differs')
    return copied


def manifest_and_audit(out, manifest_fields, inherited_audit, inherited_sha):
    # No manifest/lineage cycle: audit binds the finalized manifest, which lists
    # payloads and evidence but not itself or lineage-audit.json.
    files = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in
             ('manifest.json', 'lineage-audit.json', 'selection.sqlite')}
    manifest = dict(manifest_fields, files=files)
    write_json(out/'manifest.json', manifest)
    lineage = {k.replace('\\', '/'): v for k, v in inherited_audit['manifest_sha256'].items()}
    lineage[str(out.relative_to(ROOT)).replace('\\', '/')] = file_hash(out/'manifest.json')
    audit = {
        'passed': True, 'mechanical_checks_passed': True, 'ready_for_training': False,
        'manual_review_passed': False, 'semantic_isolation_proven': False,
        'manifest_sha256': lineage, 'data_artifact_sha256': files,
        'inherited_v5_audit': {'path': 'inherited-v5-lineage-audit.json', 'sha256': inherited_sha,
                              'passed': inherited_audit['passed']},
        'scope': 'Mechanical exact/leading96/finite stride16 lexical exclusion: new stories versus legacy v3/v5 held; new train versus selected new held; new development versus new test; selected new held versus inherited dialogue/focus plus actual teaching-curriculum-v3 teacher/foundations whole records and individual messages. Inherited unchanged payload isolation linked to v5 audit. No semantic/content approval or isolation from all historical pretraining.',
        'current_curriculum_binding': manifest_fields.get('current_curriculum_binding'),
        'new_train_does_not_overlap_new_held_under_declared_rules': True,
        'old_held_text_displayed': False, 'new_test_text_displayed': False,
        'upstream_test_accessed': False,
    }
    write_json(out/'lineage-audit.json', audit)
    # Match frozen v8's gate expressions, without importing its trainer.
    for name, expected in audit['data_artifact_sha256'].items():
        assert manifest['files'][name] == expected
    assert audit['manifest_sha256'][out.relative_to(ROOT).as_posix()] == file_hash(out/'manifest.json')
    for name in ('tokenizer.json', 'pretrain-train.bin', 'pretrain-development.bin',
                 'dialogue-train-tokens.jsonl', 'dialogue-development-tokens.jsonl', 'focus-tokens.json'):
        assert name in manifest['files']
    return manifest, audit


def prepare(args):
    started = time.perf_counter()
    script_sha = file_hash(Path(__file__))
    helper_sha = file_hash(ROOT/'scripts/prepare_p28_prose.py')
    out = output_dir(args.name)
    if out.exists() or REPORT.exists():
        raise FileExistsError('Preserve previous output/report; choose a fresh run and coordinate report handling')
    source, provenance = verified_source()
    source_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    source_provenance_sha = file_hash(MATERIAL/'provenance.json')
    inherited = json.loads((OLD/'manifest.json').read_text('utf-8'))
    inherited_audit = json.loads((OLD/'lineage-audit.json').read_text('utf-8'))
    inherited_sha = file_hash(OLD/'lineage-audit.json')
    links = {k.replace('\\', '/'): v for k,v in inherited_audit['manifest_sha256'].items()}
    if inherited_audit['passed'] is not True or links['runs/conversation-data-v5'] != file_hash(OLD/'manifest.json'):
        raise ValueError('Inherited v5 lineage is not valid for the current manifest')
    token_hash = file_hash(OLD/'tokenizer.json')
    if token_hash != inherited['tokenizer_sha256'] or token_hash != inherited['files']['tokenizer.json']:
        raise ValueError('Tokenizer mismatch')
    curriculum_manifest = json.loads((CURRICULUM/'manifest.json').read_text('utf-8'))
    curriculum_binding = {'path': CURRICULUM.relative_to(ROOT).as_posix(),
                          'manifest_sha256': file_hash(CURRICULUM/'manifest.json'),
                          'files': curriculum_manifest['files']}
    if curriculum_manifest['tokenizer_sha256'] != token_hash:
        raise ValueError('Current teaching vocabulary differs')
    for name, expected in curriculum_binding['files'].items():
        if Path(name).name != name or file_hash(CURRICULUM/name) != expected:
            raise ValueError('Current teaching artifact hash mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(OLD/'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    bos, eos = [tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>')]
    if bos is None or eos is None or tokenizer.get_vocab_size() > 65536:
        raise ValueError('Incompatible token IDs')
    pa, pq = arrow_modules()
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ['story']:
        raise ValueError('Unexpected pinned shard schema')
    old_held, old_evidence = load_old_held()
    out.mkdir(parents=True, exist_ok=False)
    copied = copy_inherited(out, inherited)
    write_json(out/'source-provenance.json', provenance)
    resume = None
    if getattr(args, 'resume_selection', None):
        previous = output_dir(args.resume_selection)
        if (previous/'manifest.json').exists():
            raise ValueError('Resume only aborted pass1 selection, never completed data')
        if file_hash(previous/'source-provenance.json') != source_provenance_sha:
            raise ValueError('Partial selection provenance differs')
        if file_hash(previous/'inherited-v5-manifest.json') != file_hash(OLD/'manifest.json'):
            raise ValueError('Partial selection inherited data differs')
        # Copy rollback journal with database, if present. SQLite recovery occurs
        # only in the fresh output, leaving aborted files untouched.
        partial_hashes = {}
        for name in ('selection.sqlite', 'selection.sqlite-journal'):
            if (previous/name).exists():
                partial_hashes[name] = file_hash(previous/name)
                shutil.copyfile(previous/name, out/name)
        resume = {'path': previous.relative_to(ROOT).as_posix(), 'files_sha256': partial_hashes,
                  'prior_filter_counts': 'Not persisted in interrupted implementation; prefix rejection reasons unavailable. Only committed eligible records reused.'}
    db = sqlite3.connect(out/'selection.sqlite')
    if resume is None:
        create_selection_db(db)
    else:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {'candidates'}
    stats = Counter()
    resume_after = -1
    if resume is not None:
        count, resume_after = db.execute('SELECT COUNT(*),MAX(row_index) FROM candidates').fetchone()
        resume_after = -1 if resume_after is None else resume_after
        stats['source_rows_scanned'] = resume_after+1
        stats['eligible_unique_stories'] = count
        stats['resumed_committed_eligible_stories'] = count
        stats['resumed_prefix_noneligible_or_duplicate_unclassified'] = resume_after+1-count
        resume.update(committed_eligible=count, resume_after_zero_based=resume_after)
        print(json.dumps({'stage':'resume_committed_selection', 'after_row':resume_after+1, 'eligible':count}), flush=True)
    output_stats = {s: Counter() for s in ('train', 'development', 'test')}
    sample, sample_rng = [], random.Random(SEED)
    split_groups = {s: set() for s in output_stats}
    split_hashes = {s: set() for s in output_stats}
    deciles = Counter()
    try:
        # Pass1 examines the complete shard. Raw TRAIN text is never repaired,
        # shortened, sentence-spliced or printed. Only eligible held text is
        # spooled; TRAIN originals are reread from parquet in pass2.
        for ordinal, text in iter_stories(parquet):
            if ordinal <= resume_after:
                continue
            stats['source_rows_scanned'] += 1
            reasons = quality_reasons(text)
            if reasons:
                stats['rejected_primary_'+reasons[0]] += 1
                continue
            ids, why = encode_whole(tokenizer, text)
            if why:
                stats['rejected_primary_'+why] += 1
                continue
            why = old_held.reason(text)
            if why:
                stats['rejected_primary_legacy_'+why] += 1
                continue
            if not add_candidate(db, text, ordinal, len(ids)):
                stats['normalized_full_duplicates'] += 1
                continue
            stats['eligible_unique_stories'] += 1
            if stats['source_rows_scanned'] % 25000 == 0:
                db.commit()
                print(json.dumps({'stage': 'pass1', 'scanned': stats['source_rows_scanned'],
                                  'eligible': stats['eligible_unique_stories']}), flush=True)
        db.commit()
        db.execute('CREATE INDEX ranked_split ON candidates(split,score)')
        print(json.dumps({'stage': 'pass1_complete', 'scanned': stats['source_rows_scanned'],
                          'eligible': stats['eligible_unique_stories']}), flush=True)
        held = ranked(db, 'test', args.held_limit)+ranked(db, 'development', args.held_limit)
        held = remove_training_conflicts(held, tokenizer, stats)
        held, new_held = separate_new_held(held, stats)
        selected = ranked(db, 'train', args.limit)+held
        selected = {r['source_row_index']: r for r in selected}
        names = {'train': 'pretrain-train', 'development': 'story-development', 'test': 'story-test'}
        with ExitStack() as stack:
            raw = {s: stack.enter_context((out/(n+'.jsonl')).open('x', encoding='utf-8')) for s,n in names.items()}
            binary = {s: stack.enter_context((out/(n+'.bin')).open('xb')) for s,n in names.items()}
            # Pass2 visits source rows in physical order, independent of selection
            # ranking. Original UTF8 text including whitespace is retained.
            for ordinal, text in iter_stories(parquet):
                metadata = selected.get(ordinal)
                if metadata is None:
                    continue
                split = metadata['split']
                assert digest(text) == metadata['source_original_sha256']
                if split == 'train':
                    why = new_held.reason(text)
                    if why:
                        stats['train_removed_new_held_'+why] += 1
                        continue
                ids, why = encode_whole(tokenizer, text)
                assert why is None and len(ids) == metadata['content_tokens']
                row = dict(metadata, text=text, source_row_1based=ordinal+1,
                    source_repository=REPOSITORY, source_revision=REVISION, source_path=SOURCE_PATH,
                    source_file_sha256=SOURCE_SHA, source_provenance_sha256=source_provenance_sha,
                    declared_license=provenance['declared_license'], text_unchanged=True)
                row = write_story(raw[split], binary[split], row, ids, bos, eos, output_stats[split]['tokens'])
                output_stats[split].update(documents=1, tokens=len(ids)+2, characters=len(text),
                                          within_document_next_token_targets=len(ids)+1)
                split_groups[split].add(row['group']); split_hashes[split].add(row['sha256'])
                deciles[f'{split}_{min(9, ordinal*10//parquet.metadata.num_rows)}'] += 1
                if split == 'train':
                    n = output_stats[split]['documents']
                    slot = n-1 if n <= 64 else sample_rng.randrange(n)
                    if slot < 64:
                        if n <= 64:
                            sample.append(row)
                        else:
                            sample[slot] = row
        for a,b in (('train','development'), ('train','test'), ('development','test')):
            assert not split_groups[a] & split_groups[b]
            assert not split_hashes[a] & split_hashes[b]
        for split, name in names.items():
            assert (out/(name+'.bin')).stat().st_size == 2*output_stats[split]['tokens']
        if not all(output_stats[s]['documents'] for s in output_stats):
            raise ValueError('An output split is empty; do not certify this preparation')
    finally:
        db.close()
    sample.sort(key=lambda r: r['source_row_index'])
    for i, row in enumerate(sample):
        row['review_index'] = i
    write_json(out/'quality-sample.json', {'split': 'train', 'seed': SEED, 'samples': sample,
        'selection': 'Algorithm-R uniform64 reservoir of final accepted train stories, sorted by source row. Not content approval.'})
    assert source_stat == (source.stat().st_size, source.stat().st_mtime_ns)
    assert file_hash(source) == SOURCE_SHA
    assert file_hash(MATERIAL/'provenance.json') == source_provenance_sha
    assert file_hash(OLD/'manifest.json') == links['runs/conversation-data-v5']
    assert file_hash(OLD/'lineage-audit.json') == inherited_sha
    assert file_hash(CURRICULUM/'manifest.json') == curriculum_binding['manifest_sha256']
    assert all(file_hash(CURRICULUM/n) == h for n,h in curriculum_binding['files'].items())
    assert file_hash(Path(__file__)) == script_sha and file_hash(ROOT/'scripts/prepare_p28_prose.py') == helper_sha
    fields = {
        'schema_version': 1, 'stage': 'prepared_pending_manual_review',
        'passed': False, 'ready_for_training': False, 'manual_review_passed': False,
        'mechanical_checks_passed': True, 'semantic_isolation_proven': False,
        'tokenizer_sha256': token_hash, 'vocabulary': tokenizer.get_vocab_size(),
        'source_provenance': provenance, 'source_provenance_sha256': source_provenance_sha,
        'inherited_v5_manifest_sha256': file_hash(OLD/'manifest.json'),
        'current_curriculum_binding': curriculum_binding,
        'inherited_files_byte_exact': copied,
        'stats': {'pretrain_train_documents': output_stats['train']['documents'],
                  'pretrain_train_tokens': output_stats['train']['tokens']},
        'story_splits': {s: dict(c) for s,c in output_stats.items()},
        'preparation_stats': dict(stats), 'source_row_decile_counts': dict(deciles),
        'resumed_selection': resume,
        'parameters': {'train_cap': args.limit, 'held_cap_per_split': args.held_limit,
                       'min_characters': 60, 'max_content_bpe_tokens': 510, 'seed': SEED},
        'split': '98/1/1 deterministic normalized-leading96 SHA256 group split, modulo1000 from first16 hex; full normalized dedup before caps. Seeded full-shard hash ranking, then cap. Test retained before lexical exclusion of development; train excluded against both. Caps may yield fewer than requested, without refilling or moving groups.',
        'story_file_names': names,
        'whole_story_policy': 'Exact complete original story including original whitespace; no truncation, repairs, inventing, or paragraph/sentence splicing. Fiction and anthropomorphism are not rejection criteria.',
        'binary_format': 'Little-endian uint16 BOS + up to510 unchanged BPE content tokens + EOS; offsets/counts stored per JSONL record. v8 draws fixed windows, so storing whole stories does not promise its individual windows remain within one story.',
        'old_held_evidence': old_evidence,
        'lexical_rule': 'Reuse HeldIndex: normalized exact, leading96, and stride16 16-character shingles with>=8 minimum count and>=80%min-size containment. Short, shifted and paraphrased overlaps may escape. No semantic proof.',
        'new_held_limits': 'Selected story held files are excluded from this new train corpus, inherited dialogue/focus, and the exact teaching-curriculum-v3 teacher/foundations under the stated rule. Checks cover whole raw records and each user/assistant message. They are not certified unseen by old pretraining, historical weights, vocabulary-fitting corpora or other ancestor data; the old pretrain TRAIN stream was not exhaustively compared. Any different curriculum needs a new audit.',
        'manual_gate': 'No training authorized. Main/Hegel must review content and the new fixed64train reservoir; the separate32source-sample review does not automatically approve this selected corpus. This preparer never sets readiness true.',
        'constraints': {'gpu': False, 'training': False, 'upstream_test_accessed': False,
                        'held_text_displayed': False, 'source_repaired': False, 'arrow_threads': 1,
                        'maximum_process_cpus': 2, 'pyarrow_version': pa.__version__},
        'script_sha256': script_sha,
        'helper_sha256': helper_sha,
        'tests_sha256': file_hash(ROOT/'tests/test_p28_stories.py'),
        'files_not_used_for_training': ['selection.sqlite', 'quality-sample.json', 'story-development.jsonl',
                                        'story-development.bin', 'story-test.jsonl', 'story-test.bin'],
    }
    manifest, audit = manifest_and_audit(out, fields, inherited_audit, inherited_sha)
    report = {'status': 'prepared_pending_manual_review', 'ready_for_training': False,
        'mechanical_checks_passed': True, 'output': out.relative_to(ROOT).as_posix(),
        'manifest_sha256': file_hash(out/'manifest.json'), 'lineage_audit_sha256': file_hash(out/'lineage-audit.json'),
        'source_sha256': SOURCE_SHA, 'source_rows_scanned': stats['source_rows_scanned'],
        'resumed_selection': resume,
        'current_curriculum_binding': curriculum_binding,
        'story_splits': fields['story_splits'], 'preparation_stats': dict(stats),
        'training_review_sample': {'path': (out/'quality-sample.json').relative_to(ROOT).as_posix(),
                                  'count': len(sample), 'sha256': file_hash(out/'quality-sample.json')},
        'v8_manifest_contract_verified': True, 'inherited_files_byte_exact': copied,
        'elapsed_seconds': time.perf_counter()-started,
        'next_invocation': '.venv/Scripts/python.exe -B scripts/prepare_p28_stories.py --name '+args.name+' --limit '+str(args.limit)+' --held-limit '+str(args.held_limit)+((' --resume-selection '+args.resume_selection) if getattr(args,'resume_selection',None) else ''),
        'invocation_note': 'Invocation above records this completed run; do not rerun against an existing directory. No trainer invocation is authorized. Main owns plan/recipe and content decision.',
        'output_contract': {'train': 'pretrain-train.bin/jsonl', 'new_story_development': 'story-development.bin/jsonl',
                            'new_story_test': 'story-test.bin/jsonl (never displayed)',
                            'legacy_held_dialogue_focus_tokenizer': 'byte-identical original filenames',
                            'lineage': 'lineage-audit.json passed=true means mechanical checks only; manifest and audit ready_for_training=false'},
        'limits': [fields['new_held_limits'], fields['lexical_rule'], fields['manual_gate'], fields['binary_format']],
    }
    write_json(REPORT, report)
    print(json.dumps({'output': report['output'], 'story_splits': fields['story_splits'],
                      'ready_for_training': False, 'seconds': report['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='candidate-v1')
    parser.add_argument('--limit', type=int, default=100000)
    parser.add_argument('--held-limit', type=int, default=1024)
    parser.add_argument('--resume-selection', help='Reuse committed pass1 rows from an aborted sibling, into fresh --name')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100000 or not 1 <= args.held_limit <= 2048:
        parser.error('Require train cap1..100000 and held cap1..2048')
    prepare(args)


if __name__ == '__main__':
    main()
