import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from scripts.build_matched_dialogue import build
from spora.conversation_data import DialogueRows


class MatchedDialogueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='spora-pair-')
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.tokenizer = self.root / 'tokenizer.json'
        tok = Tokenizer(models.BPE(unk_token='[UNK]'))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.train_from_iterator(['甲乙丙丁戊己庚辛', '很长的文字'], trainers.BpeTrainer(
            vocab_size=270, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=['[UNK]', '<|pad|>', '<|bos|>', '<|eos|>', '<|user|>', '<|assistant|>']))
        tok.save(str(self.tokenizer))
        self.tok = tok
        self.source = self.root / 'train.jsonl'

    def write(self, rows):
        self.source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

    def rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    def test_cli_pairs_targets_trimming_and_lineage(self):
        messages = [dict(role='user' if i % 2 == 0 else 'assistant', content=s)
                    for i, s in enumerate(['甲'*30, '乙'*30, '丙', '丁', '戊', '己'])]
        self.write([dict(group='family', identity='dialogue', messages=messages, source_repository='synthetic')])
        out = self.root / 'output'
        result = subprocess.run([sys.executable, '-m', 'scripts.build_matched_dialogue', '--input', str(self.source),
                                 '--tokenizer', str(self.tokenizer), '--output', str(out), '--max-tokens', '35'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        a, b, lineage = (self.rows(out / n) for n in ('context.jsonl', 'last-user.jsonl', 'lineage.jsonl'))
        self.assertEqual(len(a), 1)
        self.assertEqual(lineage[0]['removed_messages'], 2)
        self.assertEqual(lineage[0]['target_message_index'], 5)
        self.assertEqual(lineage[0]['source_metadata']['source_repository'], 'synthetic')
        self.assertEqual([x for x, m in zip(a[0]['tokens'], a[0]['mask']) if m],
                         [x for x, m in zip(b[0]['tokens'], b[0]['mask']) if m])
        self.assertEqual(a[0]['tokens'][-1], self.tok.token_to_id('<|eos|>'))
        self.assertTrue(a[0]['mask'][-1])
        self.assertEqual(sum(a[0]['mask']), sum(b[0]['mask']))
        assistant = self.tok.token_to_id('<|assistant|>')
        self.assertEqual(sum(token == assistant and not mask for token, mask in zip(a[0]['tokens'], a[0]['mask'])), 2)
        dataset = DialogueRows(out / 'context.jsonl', 34)
        self.assertEqual(len(dataset), 1)
        dataset.close()
        self.assertEqual(json.loads((out / 'manifest.json').read_text(encoding='utf-8'))['counts']['no_history_after_trim'], 1)
        with self.assertRaises(FileExistsError):
            build(self.source, self.tokenizer, out, 35)

    def test_window_overlap_and_conflict(self):
        msgs = [dict(role='user' if i % 2 == 0 else 'assistant', content=s)
                for i, s in enumerate('甲乙丙丁戊己庚辛')]
        rows = [dict(group='g', source_turn_offset=0, messages=msgs),
                dict(group='g', source_turn_offset=2, messages=msgs[2:])]
        self.write(rows)
        manifest = build(self.source, self.tokenizer, self.root / 'out', 512)
        self.assertEqual(manifest['counts']['pairs'], 3)
        self.assertEqual(manifest['counts']['duplicate_target'], 2)
        rows[1]['messages'] = [dict(m) for m in msgs[2:]]
        rows[1]['messages'][3]['content'] = '不同'
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            build(self.source, self.tokenizer, self.root / 'other', 512)

    def test_protected_path_invalid_shape_and_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, 'Protected'):
            build(self.root / 'final-test.jsonl', self.tokenizer, self.root / 'out', 512)
        self.write([{'group': 'g', 'messages': [{'role': 'system', 'content': 'x'}]}])
        with self.assertRaisesRegex(ValueError, 'invalid message'):
            build(self.source, self.tokenizer, self.root / 'bad', 512)
        self.write([{'group': 'g', 'messages': [dict(role=r, content='甲'*10) for r in ('user', 'assistant', 'user', 'assistant')]}])
        short_length = 1 + 2 * (2 + len(self.tok.encode('甲'*10).ids))
        m = build(self.source, self.tokenizer, self.root / 'trim', short_length)
        self.assertEqual(m['counts']['no_history_after_trim'], 1)
        m = build(self.source, self.tokenizer, self.root / 'long', short_length - 1)
        self.assertEqual(m['counts']['last_pair_too_long'], 1)


if __name__ == '__main__':
    unittest.main()
