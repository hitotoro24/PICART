"""Read an sklearn DecisionTreeRegressor and look for the subtree shapes that
the stencils in stencils.py require.  A configuration is accepted only if
every threshold involved is within `tol` (relative to the node's width along
that axis) of the exact midpoint; tol=0 means exact to rounding.  Nothing is
estimated from a configuration that fails the check."""
from dataclasses import dataclass, field
import numpy as np

from .stencils import diag_four_cell, mixed_four_cell

_EPS = 1e-9  # rounding slack used when tol == 0


@dataclass
class Node:
    """One node of a fitted tree together with the box it covers.
    feature/threshold describe its split (feature = -1 for a leaf); value is
    the sample mean of y in the node; lo/hi are the box bounds per axis."""
    id: int
    depth: int
    lo: np.ndarray
    hi: np.ndarray
    feature: int          # -1 for a leaf
    threshold: float
    left: int             # child ids, -1 for a leaf
    right: int
    value: float          # sample mean of y in the node
    n: int

    @property
    def is_leaf(self):
        """True when the node has no split."""
        return self.feature < 0

    def mid(self, p):
        """Midpoint of the node's extent along axis p."""
        return (self.lo[p] + self.hi[p]) / 2

    def width(self, p):
        """Width of the node's extent along axis p."""
        return self.hi[p] - self.lo[p]


def extract_nodes(reg, domain_lo, domain_hi):
    """Walk the fitted tree from the root with the known domain as the root
    extent; every node's box is obtained by cutting its parent's box."""
    t = reg.tree_
    nodes = {}
    stack = [(0, 0, np.array(domain_lo, float), np.array(domain_hi, float))]
    while stack:
        i, depth, lo, hi = stack.pop()
        f, tau = int(t.feature[i]), float(t.threshold[i])
        L, R = int(t.children_left[i]), int(t.children_right[i])
        leaf = L == -1
        nodes[i] = Node(i, depth, lo, hi, -1 if leaf else f, tau, L, R,
                        float(t.value[i].ravel()[0]), int(t.n_node_samples[i]))
        if not leaf:
            hl, lr = hi.copy(), lo.copy()
            hl[f], lr[f] = tau, tau
            stack.append((L, depth + 1, lo, hl))
            stack.append((R, depth + 1, lr, hi))
    return nodes


def split_table(nodes, d):
    """counts[depth, feature] = number of internal nodes at that depth
    splitting on that feature."""
    K = max(n.depth for n in nodes.values()) + 1
    counts = np.zeros((K, d), dtype=int)
    for n in nodes.values():
        if not n.is_leaf:
            counts[n.depth, n.feature] += 1
    return counts


def midpoint_deviation(node):
    """|tau - midpoint| / width along the split axis (0 = exact midpoint)."""
    return abs(node.threshold - node.mid(node.feature)) / node.width(node.feature)


def _at_midpoint(node, tol):
    """True when the node's threshold is within tol of its midpoint (tol=0 means exact)."""
    return midpoint_deviation(node) <= max(tol, _EPS)


@dataclass
class Config:
    """One accepted subtree configuration: the Hessian entry it estimates, the
    node at its root, the estimate, the midpoint deviations of the thresholds
    used, and the sample counts of the four cells."""
    entry: tuple
    node: int
    estimate: float
    deviations: tuple           # relative midpoint deviations of the thresholds used
    n_cells: tuple = field(default_factory=tuple)


def find_diag_configs(nodes, i, tol=0.0):
    """Node splits on i, both children split on i, all three at midpoints
    (within tol).  Four equal cells -> diag_four_cell."""
    out = []
    for n in nodes.values():
        if n.is_leaf or n.feature != i:
            continue
        L, R = nodes[n.left], nodes[n.right]
        if L.is_leaf or R.is_leaf or L.feature != i or R.feature != i:
            continue
        devs = (midpoint_deviation(n), midpoint_deviation(L), midpoint_deviation(R))
        if not all(_at_midpoint(m, tol) for m in (n, L, R)):
            continue
        LL, LR, RL, RR = (nodes[L.left], nodes[L.right], nodes[R.left], nodes[R.right])
        est = diag_four_cell(LL.value, LR.value, RL.value, RR.value, n.lo[i], n.hi[i])
        out.append(Config((i, i), n.id, est, devs, (LL.n, LR.n, RL.n, RR.n)))
    return out


def find_mixed_configs(nodes, i, j, tol=0.0):
    """Node splits on j at its midpoint; both children split on i at the
    midpoint of the (shared) i extent, hence at the same threshold.  2x2
    equal cells -> mixed_four_cell.  Both orders (split j then i, split i
    then j) are searched since either yields Q_ij."""
    out = []
    for (a, b) in ((i, j), (j, i)):           # a = inner split var, b = outer
        for n in nodes.values():
            if n.is_leaf or n.feature != b:
                continue
            B, T = nodes[n.left], nodes[n.right]
            if B.is_leaf or T.is_leaf or B.feature != a or T.feature != a:
                continue
            devs = (midpoint_deviation(n), midpoint_deviation(B), midpoint_deviation(T))
            if not all(_at_midpoint(m, tol) for m in (n, B, T)):
                continue
            BL, BR, TL, TR = (nodes[B.left], nodes[B.right], nodes[T.left], nodes[T.right])
            est = mixed_four_cell(BL.value, BR.value, TL.value, TR.value,
                                  n.lo[a], n.hi[a], n.lo[b], n.hi[b])
            out.append(Config((i, j), n.id, est, devs, (BL.n, BR.n, TL.n, TR.n)))
    return out


def assemble(configs, d):
    """Q_hat[i, j] = unweighted mean of all accepted estimates of entry (i, j);
    NaN where no configuration was accepted.  Symmetric by construction.
    (Equal weighting is provisional; see report, open choice 3.)"""
    Q_hat = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(i, d):
            vals = [c.estimate for c in configs if c.entry == (i, j)]
            if vals:
                Q_hat[i, j] = Q_hat[j, i] = float(np.mean(vals))
    return Q_hat


def extract_nodes_from_cart(root, domain_lo, domain_hi):
    """Same as extract_nodes, for Tony's from-scratch CARTRegressor
    (regression_cart_learning_workspace/cart_regressor.py): a fitted `root_`
    Node with fields prediction, n_samples, feature_index, threshold, left,
    right."""
    nodes, counter = {}, [0]

    def walk(n, depth, lo, hi):
        """Recursively convert one CARTRegressor node (and its subtree) into Node records."""
        nid = counter[0]; counter[0] += 1
        leaf = n.is_leaf
        node = Node(nid, depth, lo, hi, -1 if leaf else int(n.feature_index),
                    0.0 if leaf else float(n.threshold), -1, -1,
                    float(n.prediction), int(n.n_samples))
        nodes[nid] = node
        if not leaf:
            f, tau = node.feature, node.threshold
            hl, lr = hi.copy(), lo.copy(); hl[f] = lr[f] = tau
            node.left = walk(n.left, depth + 1, lo, hl)
            node.right = walk(n.right, depth + 1, lr, hi)
        return nid
    walk(root, 0, np.array(domain_lo, float), np.array(domain_hi, float))
    return nodes
