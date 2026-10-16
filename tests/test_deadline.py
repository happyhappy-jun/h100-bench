import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from benchkit.deadline import run


@unittest.skipUnless(sys.platform == 'linux', 'Linux child-subreaper contract')
class DeadlineTests(unittest.TestCase):
    def assert_stopped(self, pid):
        import psutil
        try:
            self.assertEqual(psutil.Process(pid).status(), psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            pass

    def test_normal_completion_cleans_detached_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = ('import subprocess,sys; from pathlib import Path; '
                      'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); '
                      f'Path({str(root / "pid")!r}).write_text(str(p.pid)); print("completed",flush=True)')
            result = run([sys.executable, '-c', script], root, os.environ, root/'case.log', 5, 1)
            self.assertEqual(result.returncode, 0, result)
            self.assertFalse(result.timed_out)
            self.assertEqual(result.cleanup_errors, [])
            self.assert_stopped(int((root/'pid').read_text()))
            self.assertIn('completed', (root/'case.log').read_text())

    def test_timeout_cleans_double_fork_and_separate_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = '\n'.join([
                'import os,time,signal', 'from pathlib import Path',
                'signal.signal(signal.SIGTERM,signal.SIG_IGN)',
                'if os.fork()==0:',
                ' os.setsid()',
                ' if os.fork()!=0: os._exit(0)',
                f' Path({str(root / "pid")!r}).write_text(str(os.getpid()))',
                ' time.sleep(30)',
                'else: time.sleep(30)',
            ])
            result = run([sys.executable, '-c', script], root, os.environ, root/'case.log', 2, .8)
            self.assertEqual(result.returncode, 124, result)
            self.assertTrue(result.timed_out)
            self.assertLess(result.elapsed, 3.5)
            self.assertEqual(result.cleanup_errors, [])
            self.assert_stopped(int((root/'pid').read_text()))
            self.assertTrue(Path(str(root/'case.log')+'.deadline.json').exists())

    def test_keyboard_interrupt_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = ('import os,time; from pathlib import Path; '
                      f'Path({str(root / "pid")!r}).write_text(str(os.getpid())); time.sleep(30)')
            driver = '\n'.join([
                'import os,signal,sys,threading',
                'from benchkit.deadline import run',
                'threading.Timer(1.0, lambda: os.kill(os.getpid(),signal.SIGINT)).start()',
                'try:',
                f' run([sys.executable,"-c",{native!r}],{directory!r},os.environ,{str(root / "case.log")!r},20,1)',
                'except KeyboardInterrupt: print("interrupted",flush=True)',
            ])
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            outcome = subprocess.run([sys.executable, '-c', driver], env=env, text=True, capture_output=True, timeout=8)
            self.assertEqual(outcome.returncode, 0, outcome.stderr)
            self.assertIn('interrupted', outcome.stdout)
            self.assert_stopped(int((root/'pid').read_text()))
            self.assertEqual(json.loads(Path(str(root/'case.log')+'.deadline.json').read_text())['returncode'], 130)

    def test_unrelated_sibling_is_not_signaled(self):
        sibling = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = run([sys.executable, '-c', 'print(42)'], directory, os.environ, Path(directory)/'case.log', 3, 1)
                self.assertEqual(result.returncode, 0, result)
                self.assertIsNone(sibling.poll())
        finally:
            sibling.terminate()
            sibling.wait(timeout=3)

    def test_positive_budget_required(self):
        with self.assertRaises(ValueError):
            run(['unused'], None, None, 'unused.log', 0)


if __name__ == '__main__':
    unittest.main()
