"""Cold subprocess validation: poison torch must never be imported by preflight."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'tools/run_local.py'


class ColdCLI(unittest.TestCase):
    def test_help_and_invalid_inputs_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'torch-imported'
            (root/'torch.py').write_text(f"from pathlib import Path; Path({str(marker)!r}).touch(); raise RuntimeError('torch imported')")
            bundle = root/'bundle'; bundle.mkdir()
            (bundle/'local-inputs.json').write_text(json.dumps(dict(files={},study_files=[],workspace={},roles={})))
            env = {**os.environ, 'PYTHONPATH':str(root), 'PYTHONDONTWRITEBYTECODE':'1'}
            help_result = subprocess.run([sys.executable,'-B',str(CLI),'--help'],capture_output=True,text=True,env=env,timeout=10)
            self.assertEqual(help_result.returncode,0)
            for study in ('physion','geometry','final256'):
                out = root / ('out-'+study)
                command = [sys.executable,'-B',str(CLI),study,'--inputs',str(bundle),'--output',str(out),'--check-inputs']
                result = subprocess.run(command,capture_output=True,text=True,env=env,timeout=20)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Required workspace',result.stderr)
                self.assertFalse(out.exists())
            self.assertFalse(marker.exists())

    def test_timeout_and_removed_child_bypass(self):
        for extra in (['--timeout','nan'], ['--timeout','inf'], ['--timeout','28801'], ['--child']):
            command = [sys.executable,'-B',str(CLI),'geometry','--inputs','missing','--output','missing-output',*extra]
            result = subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('torch',result.stderr)


if __name__ == '__main__': unittest.main()
