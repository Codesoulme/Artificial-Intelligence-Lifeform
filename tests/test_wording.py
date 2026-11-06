import unittest
import torch
from scripts.wording_task import TRAIN, DEVELOPMENT, TEST, prompt, batch, stress_rows
from scripts.ladder_task import tasks, evidence_positions
from spora.language_bridge import encode, PAD

class WordingTests(unittest.TestCase):
    def test_split_and_byte_alignment(self):
        groups=[set(TRAIN),set(DEVELOPMENT),set(TEST)]
        self.assertEqual([len(g) for g in groups],[12,2,2])
        self.assertEqual(len(set.union(*groups)),16)
        row=tasks()['update']['train'][0]
        texts=[{prompt(row,w) for w in g} for g in groups]
        self.assertFalse(texts[0]&texts[1] or texts[0]&texts[2] or texts[1]&texts[2])
        rows=[row]*16;wordings=list(TRAIN+DEVELOPMENT+TEST)
        tokens,starts=batch(rows,wordings)
        for i,w in enumerate(wordings):
            prefix=encode(prompt(row,w))[:-1]
            self.assertEqual(tokens[i,:starts[i]].tolist(),prefix)
            self.assertEqual(tokens[i,starts[i]:][tokens[i,starts[i]:]!=PAD].tolist(),encode(row['answer']+'。')[1:])
            pos=evidence_positions([row])[0]
            self.assertEqual(bytes(tokens[i,pos-2:pos+1].tolist()).decode('utf-8'),row['answer'])

    def test_stress_labels_and_long_query_blind_spot(self):
        diagnostics=stress_rows()
        for name,rows in diagnostics.items():
            for row in rows:
                expected=next((p for o,p in reversed(row['events']) if o==row['target']),'未知')
                self.assertEqual(expected,row['answer'])
                self.assertLess(len(encode(prompt(row,(0,0))))+16,256)
        row=diagnostics['long_question'][0]
        first=encode(prompt(row,'请问:{target}经过这些移动之后现在位于什么地方?'))[:-1]
        changed=dict(row,target=next(o for o in row['pair'] if o!=row['target']))
        second=encode(prompt(changed,'请问:{target}经过这些移动之后现在位于什么地方?'))[:-1]
        self.assertNotEqual(first,second)
        self.assertEqual(first[-24:],second[-24:])
        self.assertLess(len(first)+16,256)

if __name__=='__main__':unittest.main()
