import tempfile,unittest,json,hashlib
from pathlib import Path
from spora.text_data import read_manifest,windows

class TextDataTests(unittest.TestCase):
    def make(self,root):
        rows=[]
        for name,text in [('train','ab cd'),('development','ef gh'),('test','ij kl')]:
            file=root/(name+'.txt');file.write_text(text,encoding='utf-8')
            rows.append({'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'split':name,'source':'synthetic test','license':'test fixture'})
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'schema':'spora-text-v1','documents':rows}));return manifest,rows
    def test_splits_are_document_level_and_padding_is_finite(self):
        with tempfile.TemporaryDirectory() as d:
            p,rows=self.make(Path(d));documents=read_manifest(p)
            self.assertEqual(windows(documents,'train',8).shape,(1,8))
    def test_duplicate_across_splits_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,rows=self.make(Path(d));rows[1]=dict(rows[0],split='development')
            p.write_text(json.dumps({'schema':'spora-text-v1','documents':rows}))
            with self.assertRaises(ValueError):read_manifest(p)
    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=self.make(Path(d));(Path(d)/'train.txt').write_text('changed')
            with self.assertRaises(ValueError):read_manifest(p)

if __name__=='__main__':unittest.main()
