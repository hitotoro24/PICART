"""Algorithm 1: row-wise Hessian from a fitted CART regression tree.

Uses empirical child-node mean responses and quadratic effective locations.
The final estimate is None when any row is not identified.
"""
import numpy as np


def estimate_tree_hessian(tree, X):
    """Return (symmetric_Q_or_None, raw_rows, diagnostics) from all same-axis pairs.

    X: data used to fit the tree, shape (n, d); root bounds are X.min/max.
    diagnostics per axis: (axis, split-node count, distinct pairs, rank).
    """
    t = tree.tree_
    d = X.shape[1]
    stack = [(0, X.min(0), X.max(0))]
    nodes = []
    while stack:
        node, lo, hi = stack.pop()
        left, right = int(t.children_left[node]), int(t.children_right[node])
        if left == right:
            continue
        s, tau = int(t.feature[node]), float(t.threshold[node])
        if not lo[s] < tau < hi[s]:
            raise ValueError(f"Invalid bounds at CART node {node}")
        z = (lo + hi) / 2
        z[s] = (lo[s] + tau + hi[s]) / 3
        gamma = 2 * (t.value[right].item() - t.value[left].item()) / (hi[s] - lo[s])
        nodes.append((s, z, gamma))
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[s], right_lo[s] = tau, tau
        stack.extend([(left, lo.copy(), left_hi), (right, right_lo, hi.copy())])

    raw = np.full((d, d), np.nan)
    diagnostics = []
    for s in range(d):
        chosen = [(z, g) for axis, z, g in nodes if axis == s]
        k = len(chosen)
        if k < 2:
            diagnostics.append((s + 1, k, 0, 0))
            continue
        Z, g = np.stack([v[0] for v in chosen]), np.array([v[1] for v in chosen])
        i, j = np.triu_indices(k, 1)
        V, y = Z[j] - Z[i], g[j] - g[i]
        keep = np.any(V != 0, axis=1)
        V, y = V[keep], y[keep]
        rank = np.linalg.matrix_rank(V) if len(V) else 0
        diagnostics.append((s + 1, k, len(V), int(rank)))
        if rank == d:
            raw[s] = np.linalg.lstsq(V, y, rcond=None)[0]

    Q = (raw + raw.T) / 2 if np.isfinite(raw).all() else None
    return Q, raw, diagnostics
