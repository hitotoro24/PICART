"""Uneven-split extension (tree_hessian/uneven.py) on CART trees.

Part A: CART geometry, node values replaced by EXACT box means -> isolates the
        construction from sampling error.
Part B: the same trees with their own sample means -> sampling error only.
Part C: Tony's from-scratch CARTRegressor instead of sklearn (small N; it is
        O(N^2) per node).
Run:  .venv/bin/python experiments/run_uneven.py
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tree_hessian import Quadratic, extract_nodes
from tree_hessian.tree_extract import extract_nodes_from_cart
from tree_hessian.uneven import hessian_from_tree, symmetrize
from tree_hessian.synthetic import with_exact_means

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
b = np.array([-1.0, 2.0, 0.5]); c = 0.7
f = Quadratic(Q, b, c)
LO, HI = np.zeros(3), np.ones(3)
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("Q =\n", Q)


def show(nodes, label):
    """Solve the pair system on `nodes`, print rank/conditioning per row and the
    symmetrised estimate with its errors; return it (None if unidentified)."""
    Q_raw, info = hessian_from_tree(nodes, 3)
    print("\n--- %s" % label)
    print("   per row: nodes / pairs / rank / cond:",
          {s: (v["n_nodes"], v["n_pairs"], int(v["rank"]), "%.2g" % v["cond"]) for s, v in info.items()})
    if np.isnan(Q_raw).any():
        print("   unidentified rows present (NaN); raw rows:\n", Q_raw)
        return None
    Q_sym, asym = symmetrize(Q_raw)
    print("   Q_hat (symmetrised):\n", Q_sym)
    print("   entrywise error:\n", Q_sym - Q)
    print("   Frobenius ||Q_hat - Q|| = %.4f   max|err| = %.4f   raw asymmetry = %.3g"
          % (np.linalg.norm(Q_sym - Q), np.abs(Q_sym - Q).max(), asym))
    return Q_sym


print("\n" + "=" * 78 + "\nPART A: CART geometry with EXACT box means (no sampling error)\n" + "=" * 78)
X, y = f.sample_uniform(LO, HI, 20_000, np.random.default_rng(0))
for depth in (2, 3, 5, 7):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
    show(with_exact_means(extract_nodes(reg, LO, HI), f.box_mean_exact), "max_depth=%d, exact means" % depth)

print("\n" + "=" * 78 + "\nPART B: CART with its own sample means (sampling error only)\n" + "=" * 78)
print("B1  N = 20000, noiseless, by depth")
for depth in (3, 5, 7, 10):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
    show(extract_nodes(reg, LO, HI), "max_depth=%d, N=20000, sample means" % depth)

print("\nB2  Frobenius error over 10 seeds, by N and depth (noiseless unless stated)")
print("   noise   depth        N   mean Frob   sd Frob   mean max|err|   seeds unidentified")
for noise_sd in (0.0, 0.1):
    for depth in (4, 6, 8):
        for N in (2_000, 20_000, 200_000):
            fro, mx, bad = [], [], 0
            for seed in range(10):
                Xs, ys = f.sample_uniform(LO, HI, N, np.random.default_rng(seed), noise_sd)
                reg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xs, ys)
                Q_raw, info = hessian_from_tree(extract_nodes(reg, LO, HI), 3)
                if np.isnan(Q_raw).any():
                    bad += 1; continue
                Q_sym, _ = symmetrize(Q_raw)
                fro.append(np.linalg.norm(Q_sym - Q)); mx.append(np.abs(Q_sym - Q).max())
            if fro:
                print("   %4.1f   %5d   %7d   %9.4f   %7.4f   %13.4f   %d/10"
                      % (noise_sd, depth, N, np.mean(fro), np.std(fro), np.mean(mx), bad))
            else:
                print("   %4.1f   %5d   %7d   all seeds unidentified" % (noise_sd, depth, N))

print("\n" + "=" * 78 + "\nPART C: Tony's CARTRegressor (cart_regressor.py), N = 2000\n" + "=" * 78)
sys.path.insert(0, "/Users/tonys/Desktop/Class/Decision Trees/regression_cart_learning_workspace")
try:
    from cart_regressor import CARTRegressor
    Xc, yc = f.sample_uniform(LO, HI, 2_000, np.random.default_rng(0))
    for depth in (4, 6):
        t0 = time.time()
        own = CARTRegressor(max_depth=depth).fit(Xc, yc)
        t1 = time.time() - t0
        skl = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xc, yc)
        n_own = extract_nodes_from_cart(own.root_, LO, HI)
        n_skl = extract_nodes(skl, LO, HI)
        same = sorted((n.depth, n.feature, round(n.threshold, 5)) for n in n_own.values() if not n.is_leaf) == \
               sorted((n.depth, n.feature, round(float(np.float32(n.threshold)), 5)) for n in n_skl.values() if not n.is_leaf)
        print("\n   depth %d: fit %.1fs; split (depth, var, threshold) multiset identical to sklearn: %s" % (depth, t1, same))
        show(n_own, "Tony's CART, max_depth=%d, sample means" % depth)
        show(with_exact_means(n_own, f.box_mean_exact), "Tony's CART, max_depth=%d, exact means" % depth)
except ImportError as e:
    print("   cart_regressor.py not importable:", e)
