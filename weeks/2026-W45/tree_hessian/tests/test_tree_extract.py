import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from tree_hessian import (Quadratic, extract_nodes, find_diag_configs,
                          find_mixed_configs, assemble, midpoint_deviation)
from tree_hessian.tree_extract import Node

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
f = Quadratic(Q, [-1.0, 2.0, 0.5], 0.7)
LO, HI = np.zeros(3), np.ones(3)


def _ideal_tree(order, taus=None):
    """Hand-built 3-level tree with exact box means.  `order` = (root var,
    child var); thresholds default to midpoints."""
    nodes, counter = {}, [0]

    def make(lo, hi, depth):
        """Recursively build the node for box [lo, hi] at the given depth."""
        nid = counter[0]; counter[0] += 1
        if depth == 2:
            nodes[nid] = Node(nid, depth, lo, hi, -1, 0.0, -1, -1,
                              f.box_mean_exact(lo, hi), 1)
            return nid
        p = order[depth]
        tau = (lo[p] + hi[p]) / 2 if taus is None else taus[nid]
        hl, lr = hi.copy(), lo.copy(); hl[p] = lr[p] = tau
        node = Node(nid, depth, lo, hi, p, tau, -1, -1, f.box_mean_exact(lo, hi), 1)
        nodes[nid] = node
        node.left = make(lo, hl, depth + 1)
        node.right = make(lr, hi, depth + 1)
        return nid
    make(LO.copy(), HI.copy(), 0)
    return nodes


def test_ideal_diag_config_recovers_Q_ii():
    """A hand-built midpoint tree splitting x_i twice yields exactly one diag config equal to Q_ii."""
    for i in range(3):
        cfg = find_diag_configs(_ideal_tree((i, i)), i)
        assert len(cfg) == 1 and cfg[0].estimate == pytest.approx(Q[i, i], abs=1e-9)


def test_ideal_mixed_config_recovers_Q_ij():
    """A hand-built midpoint tree (split j then i, or i then j) yields one mixed config equal to Q_ij."""
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cfg = find_mixed_configs(_ideal_tree((j, i)), i, j)   # split j, then i
        assert len(cfg) == 1 and cfg[0].estimate == pytest.approx(Q[i, j], abs=1e-9)
        cfg = find_mixed_configs(_ideal_tree((i, j)), i, j)   # split i, then j
        assert len(cfg) == 1 and cfg[0].estimate == pytest.approx(Q[i, j], abs=1e-9)


def test_off_midpoint_config_is_rejected_at_tol_zero():
    """A child threshold 3% off its midpoint is rejected at tol=0, accepted at tol=0.1, and then biased."""
    taus = {0: 0.5, 1: 0.25 + 0.03, 4: 0.75}          # left child off by 3% of its width... 
    nodes = _ideal_tree((0, 0), taus)
    assert find_diag_configs(nodes, 0, tol=0.0) == []
    assert len(find_diag_configs(nodes, 0, tol=0.10)) == 1   # accepted only when tolerated
    est = find_diag_configs(nodes, 0, tol=0.10)[0].estimate
    assert abs(est - Q[0, 0]) > 1e-3                          # and then it is biased


def test_wrong_subtree_shape_yields_nothing():
    # root splits x0, children split x1: no diag config for any i, and the
    # mixed (0,1) config exists but (0,2), (1,2) do not.
    """Root on x0 with children on x1 gives no diag config and only the (0,1) mixed config; assemble leaves the rest NaN."""
    nodes = _ideal_tree((0, 1))
    assert all(find_diag_configs(nodes, i) == [] for i in range(3))
    assert len(find_mixed_configs(nodes, 0, 1)) == 1
    assert find_mixed_configs(nodes, 0, 2) == [] and find_mixed_configs(nodes, 1, 2) == []
    Q_hat = assemble(find_mixed_configs(nodes, 0, 1), 3)
    assert np.isnan(Q_hat).sum() == 7 and Q_hat[0, 1] == pytest.approx(Q[0, 1])


def test_extract_nodes_boxes_partition_domain():
    """Leaf boxes of a fitted sklearn tree tile [0,1]^3, contain exactly their samples, and store the sample mean."""
    rng = np.random.default_rng(1)
    X, y = f.sample_uniform(LO, HI, 5000, rng)
    reg = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    nodes = extract_nodes(reg, LO, HI)
    leaves = [n for n in nodes.values() if n.is_leaf]
    vol = sum(np.prod(n.hi - n.lo) for n in leaves)
    assert vol == pytest.approx(1.0)
    # every sample lands in exactly one leaf box, and that leaf's value is the
    # sample mean there
    for n in leaves:
        m = np.all((X > n.lo) & (X <= n.hi), axis=1)
        assert m.sum() == n.n and y[m].mean() == pytest.approx(n.value, rel=1e-6)
    for n in nodes.values():
        if not n.is_leaf:
            assert 0 <= midpoint_deviation(n) <= 0.5
