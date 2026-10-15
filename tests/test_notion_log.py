import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.notion_push_log import construct_record


class TestNotionLog(unittest.TestCase):
    def test_code_push_does_not_reuse_old_results(self):
        title, blocks, experiment = construct_record('a'*40, 'owner/repo',
            'Add comments', ['README.md'])
        self.assertIsNone(experiment)
        self.assertIn('Code change', title)
        self.assertIn('no new experiment', str(blocks[-1]).lower())

    def test_new_summary_creates_experiment_details(self):
        cwd=os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path('experiments').mkdir()
                Path('experiments/latest.json').write_text(json.dumps({'experiment_id':'test-run', 'description':'demo', 'figures':[]}))
                title, blocks, experiment = construct_record('b'*40, 'owner/repo',
                    'Run 1', ['experiments/latest.json'])
                self.assertIsNotNone(experiment)
                self.assertIn('Experiment', title)
                self.assertIn('test-run', str(blocks))
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
