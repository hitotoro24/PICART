import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from picart_experiment import fit_picart_stump, solve_child_predictions


class TestFirstOrderPICART(unittest.TestCase):
    def test_zero_penalty_recovers_means(self):
        assert solve_child_predictions(3, 7, 1.5, 5.0, 1.0, 8, 0) == (1.5, 5.0)

    def test_weighted_mean_preserved(self):
        nl, nr = 3, 7
        left, right = solve_child_predictions(nl, nr, 1.5, 5.0, 1.0, 8, 12)
        self.assertAlmostEqual(nl*left + nr*right, nl*1.5 + nr*5.0, places=10)

    def test_fixed_split_residual_shrinks(self):
        initial = 2 * (5.0-1.5) - 1.0
        l, r = solve_child_predictions(3, 7, 1.5, 5, 1, 1, 10)
        self.assertLess(abs(2*(r-l)-1), abs(initial))

    def test_matches_sklearn_stump_at_zero_penalty(self):
        x = np.linspace(0.01, .99, 30)
        y = np.exp(3*x)
        a = fit_picart_stump(x, y, 0.0, 3.0)
        baseline = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x[:, None], y)
        self.assertAlmostEqual(a['threshold'], float(baseline.tree_.threshold[0]), places=5)
        preds = baseline.predict(x[:, None])
        self.assertTrue(np.allclose(np.where(x<a['threshold'], a['mu_left'], a['mu_right']), preds))


if __name__ == '__main__':
    unittest.main()
