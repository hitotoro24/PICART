"""Fit sklearn CART trees to data from the 3D quadratic and ask whether the
subtree shapes required by the four-cell stencils exist.

Acceptance rule: every threshold in a configuration must lie within `tol`
(relative to the node's width on that axis) of the exact midpoint.  tol = 0
is the specified construction (aligned, equally spaced cells).  Larger tol
is reported as a DIAGNOSTIC only: those estimates use the equal-width formula
on cells that are not equal, and Part B of run_controlled.py shows the bias.
Run:  .venv/bin/python experiments/run_cart.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from tree_hessian import (Quadratic, extract_nodes, split_table, midpoint_deviation,
                          find_diag_configs, find_mixed_configs, assemble)

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
b = np.array([-1.0, 2.0, 0.5]); c = 0.7
f = Quadratic(Q, b, c)
LO, HI = np.zeros(3), np.ones(3)
PAIRS = [(0, 1), (0, 2), (1, 2)]
TOLS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def report(reg, label):
    """Print split structure, midpoint deviations, available subtree shapes, and
    the accepted configurations and estimates at each tolerance for one tree."""
    nodes = extract_nodes(reg, LO, HI)
    internal = [n for n in nodes.values() if not n.is_leaf]
    print("\n" + "=" * 78 + "\n" + label + "\n" + "=" * 78)
    print("internal nodes: %d   leaves: %d   depth: %d" % (len(internal), len(nodes) - len(internal), reg.get_depth()))
    print("splits per depth (rows) and feature x0,x1,x2 (cols):\n", split_table(nodes, 3))
    dev = np.array([midpoint_deviation(n) for n in internal])
    print("relative midpoint deviation |tau-mid|/width over internal nodes:")
    print("  min %.4f   q25 %.4f   median %.4f   q75 %.4f   max %.4f   fraction < 0.01: %.3f"
          % (dev.min(), *np.percentile(dev, [25, 50, 75]), dev.max(), (dev < 0.01).mean()))
    # subtree-shape availability regardless of threshold position
    shape_diag = {i: sum(1 for n in internal if n.feature == i
                         and not nodes[n.left].is_leaf and not nodes[n.right].is_leaf
                         and nodes[n.left].feature == i and nodes[n.right].feature == i) for i in range(3)}
    shape_mixed = {}
    for i, j in PAIRS:
        k = 0
        for a, bb in ((i, j), (j, i)):
            k += sum(1 for n in internal if n.feature == bb
                     and not nodes[n.left].is_leaf and not nodes[n.right].is_leaf
                     and nodes[n.left].feature == a and nodes[n.right].feature == a)
        shape_mixed[(i, j)] = k
    one_child = {i: sum(1 for n in internal if n.feature == i and any(
        not nodes[ch].is_leaf and nodes[ch].feature == i for ch in (n.left, n.right)))
        for i in range(3)}
    print("subtree SHAPES present (ignoring where thresholds fall):")
    print("  diag  (node i -> both children i):   ", shape_diag)
    print("  diag  (node i -> at least one child i):", one_child)
    print("  mixed (node j -> both children i or node i -> both children j):", shape_mixed)
    for tol in TOLS:
        cfgs = []
        for i in range(3):
            cfgs += find_diag_configs(nodes, i, tol)
        for i, j in PAIRS:
            cfgs += find_mixed_configs(nodes, i, j, tol)
        counts = {e: sum(1 for cg in cfgs if cg.entry == e) for e in [(0, 0), (1, 1), (2, 2)] + PAIRS}
        tag = "SPECIFIED CONSTRUCTION" if tol == 0 else "DIAGNOSTIC ONLY (equal-width formula on unequal cells)"
        print("\n-- tol = %.2f  [%s]" % (tol, tag))
        print("   accepted configurations per entry:", counts)
        if not cfgs:
            print("   no entry recoverable at this tolerance"); continue
        Q_hat = assemble(cfgs, 3)
        print("   Q_hat (NaN = not recoverable):\n", Q_hat)
        print("   entrywise error Q_hat - Q:\n", Q_hat - Q)
        if np.isnan(Q_hat).any():
            print("   Frobenius error: undefined (incomplete matrix)")
        else:
            print("   Frobenius ||Q_hat - Q|| = %.4f   (||Q||_F = %.4f)" % (np.linalg.norm(Q_hat - Q), np.linalg.norm(Q)))
        for cg in cfgs[:12]:
            print("     entry %s node %4d  est %8.4f  err %8.4f  devs (%.3f,%.3f,%.3f)  cells n=%s"
                  % (cg.entry, cg.node, cg.estimate, cg.estimate - Q[cg.entry], *cg.deviations, cg.n_cells))
        if len(cfgs) > 12:
            print("     ... %d more" % (len(cfgs) - 12))
    return nodes


if __name__ == "__main__":
    print("Q =\n", Q)
    N = 20_000
    X, y = f.sample_uniform(LO, HI, N, np.random.default_rng(0))
    print("\nN = %d uniform points on [0,1]^3, noiseless y = f(x)" % N)
    for depth in (3, 5, 7):
        reg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
        nodes = report(reg, "CART max_depth=%d, noiseless, N=%d" % (depth, N))
        if depth == 3:
            print("\ntree (first three levels):\n" + export_text(reg, feature_names=["x0", "x1", "x2"], decimals=3))
    Xn, yn = f.sample_uniform(LO, HI, N, np.random.default_rng(0), noise_sd=0.1)
    reg = DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xn, yn)
    report(reg, "CART max_depth=5, noise sd=0.1, N=%d" % N)
    # a bigger sample: does the deviation from midpoints shrink?
    N2 = 200_000
    X2, y2 = f.sample_uniform(LO, HI, N2, np.random.default_rng(1))
    reg = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X2, y2)
    report(reg, "CART max_depth=5, noiseless, N=%d" % N2)
    reg = DecisionTreeRegressor(max_depth=10, random_state=0).fit(X2, y2)
    report(reg, "CART max_depth=10, noiseless, N=%d" % N2)
