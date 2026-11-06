"""Hand-built partition trees (no CART) for exactness tests: the geometry is
chosen by the caller, the node values are exact box means."""
import numpy as np
from .tree_extract import Node


def build_tree(lo, hi, depth, choose_split, mean_fn):
    """choose_split(lo, hi, depth, node_id) -> (feature, threshold).
    Every node gets value = mean_fn(lo, hi).  Returns {id: Node}."""
    nodes, counter = {}, [0]

    def make(lo, hi, d):
        """Recursively create the node for box [lo, hi] at depth d and its subtree."""
        nid = counter[0]; counter[0] += 1
        if d == depth:
            nodes[nid] = Node(nid, d, lo, hi, -1, 0.0, -1, -1, mean_fn(lo, hi), 1)
            return nid
        p, tau = choose_split(lo, hi, d, nid)
        hl, lr = hi.copy(), lo.copy(); hl[p] = lr[p] = tau
        node = Node(nid, d, lo, hi, p, tau, -1, -1, mean_fn(lo, hi), 1)
        nodes[nid] = node
        node.left = make(lo, hl, d + 1)
        node.right = make(lr, hi, d + 1)
        return nid
    make(np.array(lo, float), np.array(hi, float), 0)
    return nodes


def random_uneven_split(rng, frac=(0.2, 0.8)):
    """Random variable, threshold uniformly in the middle 60% of the extent."""
    def choose(lo, hi, depth, nid):
        """Pick a random split variable and a threshold inside the box's extent."""
        p = int(rng.integers(len(lo)))
        return p, float(lo[p] + rng.uniform(*frac) * (hi[p] - lo[p]))
    return choose


def with_exact_means(nodes, mean_fn):
    """Copy of a node dict with every value replaced by the exact box mean:
    same geometry, zero sampling error."""
    out = {}
    for i, n in nodes.items():
        m = Node(**{**n.__dict__})
        m.value = mean_fn(n.lo, n.hi)
        out[i] = m
    return out
