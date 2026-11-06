import json,random,tempfile,unittest
from pathlib import Path
from spora.conversation_data import DialogueRows

class DialogueRowsTests(unittest.TestCase):
    def test_complete_filtering_random_access_and_slices_match_reference(self):
        rows=[dict(tokens=[1,2,3],mask=[False,False,True],group='甲'),
              dict(tokens=[1,2,3,4,5,6],mask=[False,True,True,True,True,True],group='too_long'),
              dict(tokens=[1,2],mask=[False,False],group='no_targets'),
              dict(tokens=[1,7,8,9],mask=[False,False,True,True],group='乙')]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            dataset=DialogueRows(path,4);expected=[rows[0],rows[3]]
            try:
                self.assertEqual(len(dataset),2);self.assertEqual(dataset[::-1],expected[::-1])
                self.assertEqual(dataset[-1],expected[-1])
                r1=random.Random(9);r2=random.Random(9)
                self.assertEqual([r1.choice(dataset) for _ in range(30)],[r2.choice(expected) for _ in range(30)])
                with self.assertRaises(IndexError):dataset[2]
                with self.assertRaises(IndexError):dataset[-3]
                # Mutating a fetched row must not mutate subsequent reads.
                dataset[0]['tokens'].append(999);self.assertEqual(dataset[0],expected[0])
            finally:dataset.close()
    def test_rejects_misaligned_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.jsonl';path.write_text('{"tokens":[1,2],"mask":[true]}\n')
            with self.assertRaises(ValueError):DialogueRows(path,8)

if __name__=='__main__':unittest.main()
