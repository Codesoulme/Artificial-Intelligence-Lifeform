import io,sys,tempfile,unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
from spora.train_language import main

class TrainingEntrypointTests(unittest.TestCase):
    def test_readiness_cannot_accidentally_start_language_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'should-not-exist'
            args=['train_language','--individual','missing.pt','--interface','missing.pt','--manifest','missing.json','--output',str(output)]
            with patch.object(sys,'argv',args),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code,2)
            self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
