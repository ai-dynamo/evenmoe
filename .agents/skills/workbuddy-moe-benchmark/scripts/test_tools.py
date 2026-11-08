# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests; run directly, no model/data/server required."""
import hashlib
import math
import unittest

from prepare_workloads import SUBSETS, sample, split_tasks
from summarize import summarize


class ToolsTest(unittest.TestCase):
    def test_deterministic_disjoint_stratified_sampling(self):
        rows = [{'subset': s, 'task_id': f'{i:03}',
                     'prompt_sha256': hashlib.sha256(f'{s}/{i}'.encode()).hexdigest()}
                for s, (count, _, _) in SUBSETS.items() for i in range(count)]
        calibration, test = split_tasks(rows)
        calibration2, test2 = split_tasks(list(reversed(rows)))
        self.assertEqual(calibration, calibration2)
        self.assertEqual(test, test2)
        self.assertEqual(sum(map(len, calibration.values())), 52)
        for seed in (8, 28, 2026):
            draws = sample(test, seed)
            self.assertEqual(draws, sample(test2, seed))
            self.assertEqual(len(draws), 1000)
            self.assertEqual([r['index'] for r in draws], list(range(1000)))
            for subset, (_, _, count) in SUBSETS.items():
                self.assertEqual(sum(r['subset']==subset for r in draws), count)
        self.assertNotEqual(sample(test, 8), sample(test, 28))
        self.assertTrue({r['prompt_sha256'] for g in calibration.values() for r in g}.isdisjoint(
            {r['prompt_sha256'] for g in test.values() for r in g}))

    def fixture(self):
        plan = {'stage': 'prefill', 'policies': ['baseline'], 'concurrencies': [8],
                    'seeds': [8,28,2026], 'sessions_per_seed': 1, 'rounds_per_session': 3}
        rows = [{'stage': 'prefill', 'policy': 'baseline', 'concurrency': 8, 'seed': seed,
                     'session': 1, 'round': rnd, 'status': 'ok', 'ktokens_s': value}
                for seed, mid in zip(plan['seeds'], [2,4,8])
                for rnd, value in enumerate([mid, mid, 100], 1)]
        return plan, rows

    def test_hierarchical_statistics(self):
        plan, rows = self.fixture()
        r = summarize(plan, rows)['results'][0]
        self.assertEqual(r['median_ktokens_s'], 4)
        self.assertAlmostEqual(r['stddev_ktokens_s'], math.sqrt(28/3))
        self.assertEqual(r['sample_count'], 3)

    def test_incomplete_and_failed_not_aggregated(self):
        plan, rows = self.fixture()
        for changed in [rows[:-1], [dict(r, status='interference') for r in rows]]:
            result = summarize(plan, changed)
            self.assertEqual(result['status'], 'incomplete')
            self.assertNotIn('median_ktokens_s', result['results'][0])

    def test_duplicate_and_nonfinite_rejected(self):
        plan, rows = self.fixture()
        with self.assertRaises(ValueError):
            summarize(plan, rows + rows[:1])
        rows[0]['ktokens_s'] = float('nan')
        with self.assertRaises(ValueError):
            summarize(plan, rows)

    def test_single_seed_independent_sessions(self):
        plan, rows = self.fixture()
        plan.update(seeds=[8], sessions_per_seed=2)
        rows = [r for r in rows if r['seed']==8]
        rows += [dict(r, session=2, ktokens_s=6) for r in rows.copy()]
        result = summarize(plan, rows)['results'][0]
        self.assertEqual(result['median_ktokens_s'], 4)
        self.assertAlmostEqual(result['stddev_ktokens_s'], math.sqrt(8))
        self.assertEqual(result['statistic_level'], 'session medians')
        plan['sessions_per_seed'] = 1
        result = summarize(plan, rows[:3])['results'][0]
        self.assertIsNone(result['stddev_ktokens_s'])


if __name__ == '__main__':
    unittest.main()
