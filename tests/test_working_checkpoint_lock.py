import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spora.working_checkpoint_lock import lock_working_checkpoint


class WorkingCheckpointLockTests(unittest.TestCase):
    def test_other_process_cannot_acquire_until_release(self):
        code = r'''
import sys
from spora.working_checkpoint_lock import lock_working_checkpoint
try:
    with lock_working_checkpoint(sys.argv[1]):
        print('acquired')
except RuntimeError:
    sys.exit(42)
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'working.pt'
            with lock_working_checkpoint(path):
                blocked = subprocess.run([sys.executable, '-c', code, str(path)],
                                         capture_output=True, timeout=30)
                self.assertEqual(blocked.returncode, 42, blocked.stderr.decode(errors='replace'))
            released = subprocess.run([sys.executable, '-c', code, str(path)],
                                      capture_output=True, timeout=30)
            self.assertEqual(released.returncode, 0, released.stderr.decode(errors='replace'))
            self.assertIn(b'acquired', released.stdout)

    def test_terminated_writer_releases_os_lock(self):
        code = r'''
import sys, time
from spora.working_checkpoint_lock import lock_working_checkpoint
with lock_working_checkpoint(sys.argv[1]):
    print('locked', flush=True)
    time.sleep(30)
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'working.pt'
            process = subprocess.Popen([sys.executable, '-u', '-c', code, str(path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(process.stdout.readline().strip(), b'locked')
                with self.assertRaises(RuntimeError):
                    with lock_working_checkpoint(path):
                        pass
            finally:
                process.terminate()
                process.communicate(timeout=10)
            with lock_working_checkpoint(path):
                pass


if __name__ == '__main__':
    unittest.main()
