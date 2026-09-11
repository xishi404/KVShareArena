import json
import subprocess
import sys
import unittest
from pathlib import Path

from kvsharearena.data import check_manifest, load_anchors, load_queryset
from kvsharearena.schema import validate
from kvsharearena.scoring import paired, pgr, qa_f1, score_doc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/queryset'


class ReleaseTests(unittest.TestCase):
    def test_manifest(self):
        self.assertEqual(check_manifest(DATA), {})

    def test_frozen_ids_and_anchors(self):
        anchors, _ = load_anchors()
        for subset in ('qasper', 'multifieldqa_en', 'hotpotqa'):
            rows, meta = load_queryset(subset, DATA)
            self.assertEqual(len(rows), 100)
            ids = {r['_id'] for r in rows}
            self.assertEqual(len(ids), 100)
            self.assertTrue(meta['manifest_checked'])
            for ref in ('floor', 'oracle', 'naive_pos'):
                self.assertEqual(ids, set(anchors['tracks']['re'][subset][ref]))

    def test_scoring(self):
        self.assertEqual(qa_f1('The cat.', ['cat']), 1)
        self.assertEqual(qa_f1('', ['cat']), 0)
        self.assertEqual(pgr(0.1, 0.2, 0.6), -0.25)
        self.assertIsNone(pgr(0.1, 0.2, 0.2))
        result = paired({'a': 1, 'b': 0}, {'a': 1, 'b': 0}, 'same')
        self.assertEqual(result['ci95'], [0, 0])
        self.assertTrue(result['ci_contains_zero'])

    def test_real_example(self):
        doc = json.loads((ROOT / 'examples/position_alignment_hotpotqa.json').read_text())
        anchors, _ = load_anchors()
        result = score_doc(doc, anchors, 're', 'hotpotqa')
        self.assertEqual(result['n'], 100)
        self.assertAlmostEqual(result['f1_mean'], 0.2740, places=4)
        self.assertEqual(result['self_reported_f1_drift_items'], 0)
        # A historical quality-only record must not pass as a complete submission.
        self.assertTrue(validate(doc))

    def test_report_and_frames_coverage(self):
        notes = [json.loads(s) for s in (DATA / 'agent_reports/hotpotqa_notes.jsonl').read_text().splitlines()]
        questions, _ = load_queryset('hotpotqa', DATA)
        self.assertEqual({r['_id'] for r in notes}, {r['_id'] for r in questions})
        for split, n in [('test', 120), ('calibration', 60)]:
            rows = (DATA / f'frames_variant/{split}_split.jsonl').read_text().splitlines()
            self.assertEqual(len(rows) - 1, n)

    def test_cli_cpu_only(self):
        result = subprocess.run([sys.executable, '-m', 'kvsharearena.cli', '--help'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
