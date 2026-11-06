"""Uneven-split extension.  Source: docs/claude/hessian_derivation.tex
(Proposition "Effective center", eq. (master), Section "General dimension").
Chosen by Tony on 2026-11-05 as the way to handle CART thresholds.

For quadratic f and a node with extent [l, u] splitting x_s at tau:
    gamma = 2 (mu_r - mu_l) / (u_s - l_s)                 (eq. 3, unchanged)
          = d f / d x_s (c)   exactly, with
    c_s = (l_s + tau + u_s) / 3,   c_p = (l_p + u_p) / 2  (p != s).
Because grad f is affine, two nodes a, b that both split x_s satisfy
    gamma_b - gamma_a = (c_b - c_a)^T Q e_s.
Every pair gives one row; row s of Q is the least-squares solution when the
stacked displacement matrix has full column rank.  No sample-size weighting
and no locality restriction is applied (open choices)."""
import itertools
import numpy as np

from .stencils import gradient_estimate


def effective_center(node, third=1.0 / 3.0):
    """`third` exists only so a test can show the 1/2 (child-centroid
    midpoint) variant is wrong; the construction uses 1/3."""
    c = (node.lo + node.hi) / 2
    s = node.feature
    c[s] = third * (node.lo[s] + node.threshold + node.hi[s]) if third == 1.0 / 3.0 \
        else (node.lo[s] + 2 * node.threshold + node.hi[s]) / 4
    return c


def node_gradient(nodes, node):
    """Eq.-3 gradient estimate gamma of an internal node from its children's means."""
    s = node.feature
    return gradient_estimate(nodes[node.left].value, nodes[node.right].value,
                             node.lo[s], node.hi[s])


def row_system(nodes, s, third=1.0 / 3.0):
    """All pairs of internal nodes splitting x_s.  Returns V (pairs x d) and
    delta_gamma (pairs,)."""
    ns = [n for n in nodes.values() if not n.is_leaf and n.feature == s]
    C = np.array([effective_center(n, third) for n in ns])
    g = np.array([node_gradient(nodes, n) for n in ns])
    pairs = list(itertools.combinations(range(len(ns)), 2))
    if not pairs:
        d = len(next(iter(nodes.values())).lo)
        return np.zeros((0, d)), np.zeros(0), len(ns)
    a, b = np.array(pairs).T
    return C[b] - C[a], g[b] - g[a], len(ns)


def solve_row(V, dg, rcond=1e-10):
    """Least squares; returns (q, rank, condition number).  q is NaN when
    rank < d so that an unidentified row is never reported as an estimate."""
    d = V.shape[1]
    if V.shape[0] == 0:
        return np.full(d, np.nan), 0, np.inf
    q, _, rank, sv = np.linalg.lstsq(V, dg, rcond=rcond)
    cond = sv[0] / sv[-1] if rank == d and sv[-1] > 0 else np.inf
    return (q if rank == d else np.full(d, np.nan)), rank, cond


def hessian_from_tree(nodes, d, third=1.0 / 3.0):
    """Row-by-row solve.  Returns the raw (unsymmetrised) matrix and a dict of
    diagnostics per row: number of nodes, pairs, rank, condition number."""
    Q_raw = np.full((d, d), np.nan)
    info = {}
    for s in range(d):
        V, dg, k = row_system(nodes, s, third)
        q, rank, cond = solve_row(V, dg)
        Q_raw[s] = q
        info[s] = dict(n_nodes=k, n_pairs=V.shape[0], rank=rank, cond=cond)
    return Q_raw, info


def symmetrize(Q_raw):
    """1/2 (Q + Q^T); also returns the asymmetry ||Q - Q^T||_F / ||Q||_F as a
    consistency diagnostic (rows are solved independently)."""
    Q_sym = (Q_raw + Q_raw.T) / 2
    asym = np.linalg.norm(Q_raw - Q_raw.T) / np.linalg.norm(Q_raw)
    return Q_sym, asym
