"""Focused whole-story preparation checks; never opens source or final test text."""
from array import array
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_p28_stories as prep

STORY = (
    '小狐狸住在山脚的一间木屋里，每天都会提着篮子去森林寻找新鲜的果子。'
    '一天，它遇到一棵会说话的老树，老树请它帮助迷路的小鸟找到回家的方向。'
    '小狐狸沿着清澈的小溪走到山坡，看见小鸟的妈妈正在枝头着急地张望。'
    '它把小鸟送回温暖的鸟巢，大家高兴地唱起歌来，然后一起分享篮子里的果子。'
)


class StoryPreparationTests(unittest.TestCase):
    def test_fiction_and_anthropomorphism_are_accepted_without_prose_topic_gate(self):
        self.assertEqual(prep.quality_reasons(STORY), [])
        self.assertEqual(prep.quality_reasons('  \n'+STORY+'\n '), [])

    def test_malformed_unfinished_mixed_and_repeated_stories_reject(self):
        cases = [(STORY[:-1], 'unfinished_ending'),
                 (STORY+'……', 'unfinished_ending'),
                 (STORY+'\ufffd', 'malformed_character'),
                 ('<|assistant|>'+STORY, 'markup_or_control_text'),
                 (STORY+' English translation '+('Once upon a time '*20)+'.', 'mixed_language'),
                 (STORY+STORY, 'repeated_sentence'),
                 ('“'+STORY, 'unbalanced_delimiter')]
        for text, reason in cases:
            with self.subTest(reason=reason):
                self.assertIn(reason, prep.quality_reasons(text))
        self.assertEqual(prep.quality_reasons('小鸟回家。'), ['under_60_chars'])
        self.assertEqual(prep.quality_reasons(None), ['not_text'])

    def test_full_dedup_and_leading96_group_have_distinct_meanings(self):
        full, group = prep.identities(STORY)
        same_full, same_group = prep.identities(' \n'+STORY+'\n')
        other_full, other_group = prep.identities(STORY[:110]+'最后它开心地回家了。')
        self.assertEqual((full, group), (same_full, same_group))
        self.assertNotEqual(full, other_full)
        self.assertEqual(group, other_group)
        self.assertEqual(prep.split_of_group(group), prep.split_of_group(other_group))

    def test_split_boundaries_deterministic(self):
        for bucket, expected in ((0,'test'),(9,'test'),(10,'development'),(19,'development'),(20,'train'),(999,'train')):
            self.assertEqual(prep.split_of_group(f'{bucket:016x}'+'0'*48), expected)

    def test_sqlite_global_dedup_retains_first_original_and_same_group_variants(self):
        with sqlite3.connect(':memory:') as db:
            prep.create_selection_db(db)
            self.assertTrue(prep.add_candidate(db, STORY, 7, 120))
            self.assertFalse(prep.add_candidate(db, ' \n'+STORY, 19, 122))
            self.assertTrue(prep.add_candidate(db, STORY[:110]+'最后它开心地回家了。', 21, 105))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT row_index,original_sha FROM candidates WHERE full_hash=?', (prep.identities(STORY)[0],)).fetchone(), (7, prep.digest(STORY)))
            split = prep.split_of_group(prep.identities(STORY)[1])
            self.assertEqual(prep.ranked(db, split, 1), prep.ranked(db, split, 1))
            self.assertEqual(len(prep.ranked(db, split, 1)), 1)

    def test_late_held_excludes_earlier_train_without_reassigning_split(self):
        original = ''.join(chr(0x4e00+i) for i in range(320))
        shifted = ''.join(chr(0x6000+i) for i in range(16))+original[16:]
        records = [{'text': original, 'split': 'test'}, {'text': shifted, 'split': 'development'}]
        stats = Counter()
        kept, index = prep.separate_new_held(records, stats)
        self.assertEqual(len(kept), 1)
        self.assertEqual(index.reason(shifted), 'held_sampled_lexical')
        self.assertEqual(stats['development_removed_new_test_held_sampled_lexical'], 1)
        # An earlier candidate is checked only after all held candidates exist.
        self.assertIsNotNone(index.reason(original))

    def test_matching_held_identifies_entries_for_inherited_train_exclusion(self):
        original = ''.join(chr(0x4e00+i) for i in range(320))
        shifted = ''.join(chr(0x6000+i) for i in range(16))+original[16:]
        matcher = prep.MatchingHeld([{'text': original}, {'text': STORY}])
        self.assertEqual(matcher.matches(shifted), {0})
        self.assertEqual(matcher.matches(' \n'+STORY), {1})
        self.assertEqual(matcher.matches('完全不同的一句话。'), set())

    def test_actual_teacher_and_foundation_messages_exclude_new_held(self):
        records = [{'text': STORY, 'split': 'test'},
                   {'text': STORY+'小狐狸睡着了。', 'split': 'development'}]
        teacher = json.dumps({'text': '合并记录并不相同。',
                              'messages': [{'role': 'user', 'content': STORY}]}, ensure_ascii=False)+'\n'
        foundation = json.dumps({'text': '另外的合并记录。',
                                 'messages': [{'role': 'assistant', 'content': STORY+'小狐狸睡着了。'}]}, ensure_ascii=False)+'\n'
        def opening(path, *a, **kw):
            return io.StringIO({'dialogue-train.jsonl': '', 'teacher.jsonl': teacher,
                                'foundations.jsonl': foundation}[path.name])
        stats = Counter()
        tok = type('Tokenizer', (), {'decode': lambda self, seq, skip_special_tokens: ''})()
        with patch.object(Path, 'open', opening), patch.object(Path,'read_text',return_value='[]'):
            kept = prep.remove_training_conflicts(records, tok, stats)
        self.assertEqual(kept, [])
        self.assertEqual(stats['teacher_rows_scanned_for_new_held'], 1)
        self.assertEqual(stats['foundations_rows_scanned_for_new_held'], 1)
        self.assertEqual(stats['new_held_removed_current_training_overlap'], 2)

    def test_510_content_limit_never_truncates_or_repairs(self):
        class Tokenizer:
            def __init__(self, n, decoded): self.n, self.decoded = n, decoded
            def encode(self, text, add_special_tokens=False):
                self.input = text
                return type('Encoding', (), {'ids': [9]*self.n})()
            def decode(self, ids, skip_special_tokens=False): return self.decoded
        tok = Tokenizer(510, STORY)
        ids, reason = prep.encode_whole(tok, STORY)
        self.assertEqual(len(ids), 510)
        self.assertIsNone(reason)
        self.assertEqual(tok.input, STORY)
        self.assertEqual(prep.encode_whole(Tokenizer(511, STORY), STORY), (None,'over_510_content_tokens'))
        self.assertEqual(prep.encode_whole(Tokenizer(50, STORY.strip()+'改'), STORY), (None,'tokenizer_roundtrip'))

    def test_real_tokenizer_keeps_full_whitespace_and_binary_boundaries(self):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(prep.OLD/'tokenizer.json'))
        text = ' \n'+STORY+'\n '
        ids, reason = prep.encode_whole(tok, text)
        self.assertIsNone(reason)
        raw, binary = io.StringIO(), io.BytesIO()
        row = {'text': text, 'source_original_sha256': prep.digest(text), 'source_row_index': 7}
        result = prep.write_story(raw, binary, row, ids, 1, 2, 37)
        packed = array('H'); packed.frombytes(binary.getvalue())
        if sys.byteorder != 'little': packed.byteswap()
        self.assertEqual(packed.tolist(), [1]+ids+[2])
        self.assertEqual(json.loads(raw.getvalue())['text'], text)
        self.assertEqual(result['token_offset'], 37)
        self.assertEqual(result['token_count'], len(ids)+2)
        self.assertEqual(tok.decode(packed.tolist()[1:-1]), text)

    def test_parquet_iteration_is_bounded_and_row_ordinals_preserve_nulls(self):
        pa, pq = prep.arrow_modules()
        sink = pa.BufferOutputStream()
        pq.write_table(pa.table({'story': [STORY, None, '第二条。']}), sink, row_group_size=2)
        source = pq.ParquetFile(pa.BufferReader(sink.getvalue()))
        self.assertEqual(list(prep.iter_stories(source)), [(0, STORY), (1, None), (2, '第二条。')])

    def test_pinned_provenance_rejects_source_test_wrong_revision_and_escape(self):
        good = {'repository': prep.REPOSITORY, 'revision': prep.REVISION, 'path': prep.SOURCE_PATH,
                'split': 'train', 'sha256': prep.SOURCE_SHA, 'local_file': Path(prep.SOURCE_PATH).name}
        for changed in ({'split': 'test'}, {'revision':'wrong'}, {'local_file':'../outside.parquet'}, {'sha256':'0'*64}):
            with self.subTest(changed=changed), patch.object(Path,'is_file',return_value=True), \
                    patch.object(Path,'read_text',return_value=json.dumps(dict(good, **changed))):
                with self.assertRaises(ValueError): prep.verified_source()

    def test_output_cannot_escape_and_existing_artifacts_are_not_overwritten(self):
        for name in ('../old', 'D:/old', 'a/b', 'a\\b', ''):
            with self.assertRaises(ValueError): prep.output_dir(name)
        with patch.object(Path,'exists',return_value=True), patch.object(prep,'verified_source') as verifier:
            with self.assertRaises(FileExistsError):
                prep.prepare(type('Args', (), {'name': 'existing'})())
            verifier.assert_not_called()

    def test_manifest_contract_binds_new_path_and_never_implies_approval(self):
        prep.BASE.mkdir(parents=True, exist_ok=True)
        # Any filesystem fixtures are confined to the newly authorized tree.
        with tempfile.TemporaryDirectory(prefix='unit-', dir=prep.BASE) as folder:
            out = Path(folder)
            for name in ('tokenizer.json','pretrain-train.bin','pretrain-development.bin',
                         'dialogue-train-tokens.jsonl','dialogue-development-tokens.jsonl','focus-tokens.json'):
                (out/name).write_bytes(b'fixture')
            inherited = {'passed': True, 'manifest_sha256': {'runs\\conversation-data-v5': 'old-digest'}}
            fields = {'ready_for_training': False, 'passed': False, 'tokenizer_sha256': 'toy'}
            manifest, audit = prep.manifest_and_audit(out, fields, inherited, 'audit-digest')
            self.assertTrue(audit['passed'])
            self.assertFalse(audit['ready_for_training'])
            self.assertFalse(manifest['ready_for_training'])
            self.assertFalse(audit['semantic_isolation_proven'])
            self.assertEqual(audit['manifest_sha256'][out.relative_to(prep.ROOT).as_posix()], prep.file_hash(out/'manifest.json'))
            self.assertEqual(audit['data_artifact_sha256'], manifest['files'])
            self.assertNotIn('lineage-audit.json', manifest['files'])
            self.assertIn('runs/conversation-data-v5', audit['manifest_sha256'])


if __name__ == '__main__':
    unittest.main()
