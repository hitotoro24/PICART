"""Tests state the analytical value expected and check it; failure tests
assert that a violated assumption produces a *nonzero* error of the predicted
shape, not merely that code runs."""
import numpy as np
import pytest

from tree_hessian import (Quadratic, diag_three_cell, diag_four_cell,
                          mixed_four_cell, cells_diag_three, cells_diag_four,
                          cells_mixed_four)
from tree_hessian.quadratic import box_mean_sample

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
b = np.array([-1.0, 2.0, 0.5])
f = Quadratic(Q, b, 0.7)
rng = np.random.default_rng(0)


def random_box(rng, d=3):
    """A random box in R^d with side lengths between 0.1 and 1.5."""
    lo = rng.uniform(-2, 2, d)
    return lo, lo + rng.uniform(0.1, 1.5, d)


# ---- exact regional averaging ---------------------------------------------
def test_box_mean_matches_closed_form():
    # mean of a quadratic over a box = f(centroid) + sum_p Q_pp w_p^2 / 24
    """Exact box mean equals f(centroid) + sum_p Q_pp w_p^2 / 24."""
    for _ in range(20):
        lo, hi = random_box(rng)
        w = hi - lo
        closed = f((lo + hi) / 2)[0] + np.sum(np.diag(Q) * w ** 2) / 24
        assert f.box_mean_exact(lo, hi) == pytest.approx(closed, abs=1e-12)


def test_box_mean_matches_monte_carlo():
    """Exact box mean agrees with the sample mean of 400k uniform draws."""
    lo, hi = np.zeros(3), np.ones(3)
    X, y = f.sample_uniform(lo, hi, 400_000, rng)
    assert f.box_mean_exact(lo, hi) == pytest.approx(y.mean(), abs=5e-3)


# ---- diagonal entries, exact means: zero approximation error -----------------
@pytest.mark.parametrize("i", [0, 1, 2])
def test_diag_three_cell_exact(i):
    """Three equal adjacent cells with exact means give Q_ii to 1e-9."""
    for _ in range(10):
        lo, hi = random_box(rng)
        h = rng.uniform(0.01, 0.5)
        cells = cells_diag_three(lo, hi, i, h, center=rng.uniform(-1, 1))
        mu = [f.box_mean_exact(*c) for c in cells]
        assert diag_three_cell(*mu, h) == pytest.approx(Q[i, i], abs=1e-9)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_diag_four_cell_exact(i):
    """Midpoint split plus midpoint child splits with exact means give Q_ii to 1e-9."""
    for _ in range(10):
        lo, hi = random_box(rng)
        mu = [f.box_mean_exact(*c) for c in cells_diag_four(lo, hi, i)]
        assert diag_four_cell(*mu, lo[i], hi[i]) == pytest.approx(Q[i, i], abs=1e-9)


# ---- mixed entries, exact means: zero approximation error ---------------------
@pytest.mark.parametrize("i,j", [(0, 1), (0, 2), (1, 2)])
def test_mixed_four_cell_exact(i, j):
    """A 2x2 block of equal cells with exact means gives Q_ij to 1e-9."""
    for _ in range(10):
        lo, hi = random_box(rng)
        mu = [f.box_mean_exact(*c) for c in cells_mixed_four(lo, hi, i, j)]
        est = mixed_four_cell(*mu, lo[i], hi[i], lo[j], hi[j])
        assert est == pytest.approx(Q[i, j], abs=1e-9)


# ---- predicted failures when (A2) is violated --------------------------------
def test_diag_four_cell_biased_when_child_splits_uneven():
    """Child split moved off its midpoint by delta*width: the equal-width
    formula is no longer exact and the error grows linearly with delta."""
    lo, hi = np.zeros(3), np.ones(3)
    i = 0
    errs = []
    for delta in (0.05, 0.10, 0.20):
        tau_L = 0.25 + delta
        mu = [f.box_mean_exact(*c) for c in cells_diag_four(lo, hi, i, tau_L=tau_L)]
        errs.append(abs(diag_four_cell(*mu, lo[i], hi[i]) - Q[i, i]))
    assert errs[0] > 1e-3
    assert errs[1] / errs[0] == pytest.approx(2.0, rel=0.05)
    assert errs[2] / errs[0] == pytest.approx(4.0, rel=0.05)


def test_mixed_four_cell_biased_when_i_splits_misaligned():
    """tau_T != tau_B: the cross stencil picks up Q_ii and the error grows
    linearly with the misalignment."""
    lo, hi = np.zeros(3), np.ones(3)
    i, j = 0, 1
    errs = []
    for delta in (0.05, 0.10, 0.20):
        mu = [f.box_mean_exact(*c)
              for c in cells_mixed_four(lo, hi, i, j, tau_B=0.5 - delta, tau_T=0.5 + delta)]
        errs.append(abs(mixed_four_cell(*mu, lo[i], hi[i], lo[j], hi[j]) - Q[i, j]))
    assert errs[0] > 1e-3
    assert errs[1] / errs[0] == pytest.approx(2.0, rel=0.05)
    assert errs[2] / errs[0] == pytest.approx(4.0, rel=0.05)


# ---- observation (not part of the specified construction; to be derived) -----
def test_mixed_four_cell_unaffected_by_uneven_outer_split():
    """OBSERVATION: with the two i-thresholds aligned at the midpoint, moving
    the outer j-split off its midpoint leaves the mixed stencil exact.  This
    is stronger than assumption (A2) requires and has not been derived."""
    lo, hi = np.zeros(3), np.ones(3)
    i, j = 0, 1
    for tau_j in (0.2, 0.35, 0.8):
        mu = [f.box_mean_exact(*c) for c in cells_mixed_four(lo, hi, i, j, tau_j=tau_j)]
        est = mixed_four_cell(*mu, lo[i], hi[i], lo[j], hi[j])
        assert est == pytest.approx(Q[i, j], abs=1e-9)


# ---- sampling error ------------------------------------------------------------
def test_sampling_error_shrinks_with_n():
    """With sampled cell means the diag error at n=200k is under a third of that at n=2k."""
    lo, hi = np.zeros(3), np.ones(3)
    i = 0
    cells = cells_diag_four(lo, hi, i)
    err = {}
    for n in (2_000, 200_000):
        e = []
        for seed in range(5):
            X, y = f.sample_uniform(lo, hi, n, np.random.default_rng(seed))
            mu = [box_mean_sample(X, y, *c)[0] for c in cells]
            e.append(abs(diag_four_cell(*mu, lo[i], hi[i]) - Q[i, i]))
        err[n] = np.mean(e)
    assert err[200_000] < err[2_000] / 3
