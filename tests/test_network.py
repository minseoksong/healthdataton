"""Synthetic tests: never include patient source data in this repository."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from build_network import build_network


def fixture():
    return pd.DataFrame({
        'age': [40., 42., 60., 16.], 'weight_kg': [60., 62., 80., 50.],
        'sbp': [110., 112., 140., 110.], 'hr': [70., 72., 95., 70.],
        'spo2': [98., 97.9, 97., 98.], 'creatinine': [.8, .9, 1.4, .8],
        'bun': [12., 13., 30., 12.], 'sodium': [138., 139., 145., 138.],
        'potassium': [3.8, 3.9, 4.5, 3.8], 'sex': ['F', 'F', 'M', 'F'],
        'age_under18': [0, 0, 0, 1], 'diuretic_given': [1, 0, 0, 0],
        'hf_any': [0, 0, 1, 0], 'mv_pre6h': [1, 1, 0, 1],
        't0_after_icu_out': [0]*4, 't0_after_death': [0]*4,
        'subject_id': ['private-do-not-export']*4,
    })


def load_payload(out):
    html = (out/'patient_similarity_network.html').read_text()
    payload = html.split('const DATA=', 1)[1].split(';\nconst nodes=', 1)[0]
    return json.loads(payload), html


class NetworkTests(unittest.TestCase):
    def test_raw_flags_age_group_and_no_identifiers(self):
        data = fixture()
        data['creatinine_missing_outlier'] = [0, 1, 0, 0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = build_network(data, out)
            payload, html = load_payload(out)
            self.assertEqual(summary['nodes'], 4)
            self.assertFalse(payload['nodes'][1]['valid'][5])
            self.assertTrue(all(pair[0] != 3 for pair in payload['nearest'][0]))
            self.assertEqual(next(p[2] for p in payload['nearest'][0] if p[0] == 1), 8)
            self.assertNotIn('private-do-not-export', html)
            self.assertIn('id="comparison"', html)
            self.assertIn('role="tooltip"', html)

    def test_outcomes_do_not_affect_similarity(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            build_network(data, out)
            first, _ = load_payload(out)
            data['uo_post6h_ml'] = [1., 99999., 0., 123.]
            data['hospital_expire_flag'] = [1, 1, 0, 0]
            build_network(data, out)
            second, _ = load_payload(out)
            self.assertEqual(first['nearest'], second['nearest'])

    def test_insufficient_features_and_zero_edges(self):
        data = fixture().iloc[:2].copy()
        for c in ['spo2', 'creatinine', 'bun', 'sodium', 'potassium']:
            data[c] = float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = build_network(data, out)
            payload, _ = load_payload(out)
            self.assertEqual(summary['edges'], 0)
            self.assertEqual(summary['isolates'], 2)
            self.assertEqual(payload['nearest'], [[], []])
            self.assertEqual(len(pd.read_csv(out/'patient_network_edges.csv')), 0)

    def test_invalid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing required columns'):
                build_network(pd.DataFrame({'age': [1, 2]}), Path(tmp))
            with self.assertRaisesRegex(ValueError, 'zero-based row index'):
                build_network(fixture(), Path(tmp), 4)


if __name__ == '__main__':
    unittest.main()
