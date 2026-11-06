import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from tree_hessian import Quadratic, extract_nodes
from tree_hessian.uneven import (effective_center, node_gradient, hessian_from_tree,
                                 symmetrize, row_system, solve_row)
from tree_hessian.synthetic import build_tree, random_uneven_split, with_exact_means
from tree_hessian.tree_extract import Node

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
b = np.array([-1.0, 2.0, 0.5])
f = Quadratic(Q, b, 0.7)
LO, HI = np.zeros(3), np.ones(3)


def grad(x):
    """Gradient of the test quadratic, b + Q x."""
    return b + Q @ x


def test_gamma_equals_partial_at_effective_center():
    """Proposition 'Effective center': gamma = d f/d x_s at c, any threshold."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        lo = rng.uniform(-1, 1, 3); hi = lo + rng.uniform(0.1, 2, 3)
        s = int(rng.integers(3)); tau = lo[s] + rng.uniform(0.05, 0.95) * (hi[s] - lo[s])
        nodes = build_tree(lo, hi, 1, lambda *a, s=s, tau=tau: (s, tau), f.box_mean_exact)
        n = nodes[0]
        assert node_gradient(nodes, n) == pytest.approx(grad(effective_center(n))[s], abs=1e-10)
        # the child-centroid midpoint (l + 2 tau + u)/4 is NOT where gamma lives
        if abs(tau - (lo[s] + hi[s]) / 2) > 0.05 * (hi[s] - lo[s]):
            assert abs(node_gradient(nodes, n) - grad(effective_center(n, third=0.5))[s]) > 1e-4


def test_random_uneven_tree_exact_means_recovers_identified_rows():
    """Rows whose displacement matrix has full rank are exact; rows without
    are NaN, never a number.  Depth 4 trees (15 internal nodes) often leave a
    row unidentified; depth 6 trees identify every row in these seeds."""
    full = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        nodes = build_tree(LO, HI, 4, random_uneven_split(rng), f.box_mean_exact)
        Q_raw, info = hessian_from_tree(nodes, 3)
        for s in range(3):
            if info[s]["rank"] == 3:
                full += 1
                assert np.allclose(Q_raw[s], Q[s], atol=1e-8)
            else:
                assert np.isnan(Q_raw[s]).all()
    assert full >= 5
    for seed in range(3):
        rng = np.random.default_rng(seed)
        nodes = build_tree(LO, HI, 6, random_uneven_split(rng), f.box_mean_exact)
        Q_raw, info = hessian_from_tree(nodes, 3)
        assert all(info[s]["rank"] == 3 for s in range(3)), info
        Q_sym, asym = symmetrize(Q_raw)
        assert np.allclose(Q_raw, Q, atol=1e-8) and asym < 1e-9


def test_half_instead_of_third_is_biased():
    """Using (l + 2 tau + u)/4 instead of (l + tau + u)/3 as the centre gives an error above 1e-2."""
    rng = np.random.default_rng(1)
    nodes = build_tree(LO, HI, 6, random_uneven_split(rng), f.box_mean_exact)
    Q_raw, info = hessian_from_tree(nodes, 3, third=0.5)
    assert all(info[s]["rank"] == 3 for s in range(3))
    assert np.abs(Q_raw - Q).max() > 1e-2


def test_rank_deficient_row_is_reported_not_estimated():
    # depth-1 tree: one node splits x0 -> no pairs -> rank 0, NaN row, others empty
    """Too few or collinear splitting nodes give rank < d and a NaN row, never a number."""
    nodes = build_tree(LO, HI, 1, lambda lo, hi, d, i: (0, 0.4), f.box_mean_exact)
    Q_raw, info = hessian_from_tree(nodes, 3)
    assert np.isnan(Q_raw).all() and info[0]["rank"] == 0 and info[1]["n_nodes"] == 0
    # all x0-splitting nodes aligned along x0 only (depth-2 chain x0 -> x0): the
    # displacements have no x1/x2 component, so rank 1 < 3
    nodes = build_tree(LO, HI, 2, lambda lo, hi, d, i: (0, lo[0] + 0.4 * (hi[0] - lo[0])),
                       f.box_mean_exact)
    V, dg, k = row_system(nodes, 0)
    q, rank, cond = solve_row(V, dg)
    assert k == 3 and rank == 1 and np.isnan(q).all()


def test_cart_geometry_with_exact_means_is_exact():
    """A depth-5 sklearn tree's boxes with exact box means recover Q to 1e-8."""
    rng = np.random.default_rng(2)
    X, y = f.sample_uniform(LO, HI, 5000, rng)
    reg = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
    nodes = with_exact_means(extract_nodes(reg, LO, HI), f.box_mean_exact)
    Q_raw, info = hessian_from_tree(nodes, 3)
    assert all(info[s]["rank"] == 3 for s in range(3))
    assert np.allclose(Q_raw, Q, atol=1e-8)


def test_cart_sample_means_error_shrinks_with_n():
    """With the tree's own leaf means the Frobenius error at N=200k is under a third of that at N=2k."""
    errs = {}
    for N in (2_000, 200_000):
        e = []
        for seed in range(3):
            X, y = f.sample_uniform(LO, HI, N, np.random.default_rng(seed))
            reg = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
            Q_raw, info = hessian_from_tree(extract_nodes(reg, LO, HI), 3)
            Q_sym, _ = symmetrize(Q_raw)
            e.append(np.linalg.norm(Q_sym - Q))
        errs[N] = np.mean(e)
    assert errs[200_000] < errs[2_000] / 3
