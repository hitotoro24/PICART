"""Shared code for the maple_diagnostics notebooks: CART-derived Hessian (Week 45 method,
unchanged) versus an independently implemented residual-based PHD estimator on
the small OPFData IEEE-118 subset.

Column convention (headerless CSV): column 0 = response (operating cost),
columns 1..d = predictors (d = 2 or 3).  Loads are labelled "Load k" for CSV
column k.

The CART reconstruction is imported from weeks/2026-W45/tree_hessian so that
the mathematical method is the existing one.  Nothing in that package is
modified.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# locate and import the Week 45 package without modifying it
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_W45 = _HERE.parents[1] / "2026-W45"
if str(_W45) not in sys.path:
    sys.path.insert(0, str(_W45))
from tree_hessian.tree_extract import extract_nodes            # noqa: E402
from tree_hessian.uneven import (effective_center, node_gradient,  # noqa: E402
                                 row_system, solve_row, hessian_from_tree,
                                 symmetrize)
from tree_hessian.quadratic import Quadratic                    # noqa: E402

DEFAULT_CSV = Path("/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv")
FALLBACK_CSVS = (Path("118_ieee_smol.csv"),
                 Path("/mnt/user-data/uploads/118_ieee_smol.csv"))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def software_versions() -> pd.DataFrame:
    """Return a one-row table of the Python and library versions in use."""
    return pd.DataFrame([{
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }]).T.rename(columns={0: "version"})


def load_opf(d: int, path: Path = DEFAULT_CSV):
    """Read response (CSV column 0) and predictors (CSV columns 1..d).

    Inputs: d in {2, 3}; path to the headerless CSV (fallbacks tried if absent).
    Outputs: X (n, d) float array, y (n,) float array, and the path actually read.
    Only the first d+1 columns are read; the file is never modified.
    """
    if d not in (2, 3):
        raise ValueError("d must be 2 or 3")
    actual = next((p for p in (path, *FALLBACK_CSVS) if p.is_file()), None)
    if actual is None:
        raise FileNotFoundError(f"CSV not found at {path} or fallbacks")
    data = pd.read_csv(actual, header=None, usecols=range(d + 1)).to_numpy(float)
    if not np.isfinite(data).all():
        raise ValueError("non-finite values in the selected columns")
    return data[:, 1:], data[:, 0], actual


def describe_inputs(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Return per-column n, mean, std, min, max for the response and predictors."""
    cols = {"cost (col 0)": y}
    cols.update({f"Load {k+1} (col {k+1})": X[:, k] for k in range(X.shape[1])})
    df = pd.DataFrame(cols)
    out = df.describe().T[["count", "mean", "std", "min", "max"]]
    out["count"] = out["count"].astype(int)
    return out


# ---------------------------------------------------------------------------
# CART side (Week 45 method)
# ---------------------------------------------------------------------------
def fit_cart(X: np.ndarray, y: np.ndarray, max_depth: int, min_samples_leaf: int,
             random_state: int = 0) -> DecisionTreeRegressor:
    """Fit an ordinary sklearn CART regression tree (squared-error criterion).

    Inputs: X (n, d), y (n,), max_depth, min_samples_leaf, random_state.
    Output: the fitted DecisionTreeRegressor.  No custom splitting rule is used.
    """
    return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 random_state=random_state).fit(X, y)


def nodes_from_cart(tree: DecisionTreeRegressor, X: np.ndarray) -> dict:
    """Walk the fitted tree with the Week 45 extractor.

    Root extent is the observed data range [X.min(0), X.max(0)] per axis; every
    child box is its parent's box cut at the threshold.  Output: dict id -> Node.
    """
    return extract_nodes(tree, X.min(axis=0), X.max(axis=0))


def split_table(nodes: dict) -> pd.DataFrame:
    """Tabulate every internal node: depth, split axis, extent, threshold, child
    means, eq.-3 gradient estimate and effective centre.

    Input: node dict from `nodes_from_cart`.  Output: one row per internal node.
    The gradient is gamma = 2 (mean_right - mean_left) / (hi_s - lo_s), computed
    from the two child-node response means (Week 45 `node_gradient`).
    """
    rows = []
    for n in nodes.values():
        if n.is_leaf:
            continue
        s = n.feature
        c = effective_center(n)
        rows.append({
            "node": n.id, "depth": n.depth, "split axis": f"Load {s+1}",
            "lo_s": n.lo[s], "tau": n.threshold, "hi_s": n.hi[s],
            "n": n.n, "n_left": nodes[n.left].n, "n_right": nodes[n.right].n,
            "mean_left": nodes[n.left].value, "mean_right": nodes[n.right].value,
            "gamma (eq. 3)": node_gradient(nodes, n),
            **{f"c_{k+1}": c[k] for k in range(len(c))},
        })
    return pd.DataFrame(rows).sort_values(["split axis", "depth", "node"]).reset_index(drop=True)


def cart_hessian_raw(nodes: dict, d: int):
    """Row-wise reconstruction with the Week 45 solver.

    Inputs: node dict, dimension d.  Outputs: raw (unsymmetrised) d x d matrix
    with NaN rows where rank < d, and a diagnostics table (nodes, pairs, rank,
    condition number) per row.  Row i uses only nodes that split axis i.
    """
    Q_raw, info = hessian_from_tree(nodes, d)
    diag = pd.DataFrame([{"row": f"Load {s+1}", **info[s]} for s in range(d)])
    return Q_raw, diag


def centred_regression_rows(nodes: dict, d: int) -> np.ndarray:
    """Alternative solve used by cedar_baseline: regress gamma on the effective
    centres after subtracting means, one regression per axis.

    Inputs: node dict, d.  Output: d x d raw matrix (NaN where rank < d).
    Used only to check numerically that it agrees with the pairwise system of
    the Week 45 solver; it is not the reported estimator.
    """
    out = np.full((d, d), np.nan)
    for s in range(d):
        ns = [n for n in nodes.values() if not n.is_leaf and n.feature == s]
        if len(ns) < d + 1:
            continue
        C = np.array([effective_center(n) for n in ns])
        g = np.array([node_gradient(nodes, n) for n in ns])
        V, dg = C - C.mean(0), g - g.mean()
        q, _, rank, _ = np.linalg.lstsq(V, dg, rcond=1e-10)
        if rank == d:
            out[s] = q
    return out


# ---------------------------------------------------------------------------
# PHD side (independent implementation)
# ---------------------------------------------------------------------------
def phd_residual(X: np.ndarray, y: np.ndarray):
    """Residual-based principal Hessian directions (Li 1992), original units.

    Conventions:
      * centring: Xc = X - mean(X); Sigma = Xc^T Xc / n   (1/n, not 1/(n-1));
      * residual: r = y - fitted values of the ordinary least-squares fit of y
        on [1, X] (intercept plus linear terms);
      * moment:   M = sum_i r_i Xc_i Xc_i^T / n;
      * estimate: H_phd = Sigma^{-1} M Sigma^{-1}.
    Under a Gaussian design, E[r x x^T] identifies the average Hessian of
    E[y|x]; no such assumption is checked here.  Inputs: X (n, d), y (n,).
    Outputs: H_phd (d x d, as computed, not symmetrised), condition number of
    Sigma, and the Frobenius asymmetry of H_phd (rounding-level if the
    formula is implemented symmetrically).
    """
    n = len(y)
    Xc = X - X.mean(axis=0)
    Sigma = Xc.T @ Xc / n
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    r = y - design @ beta
    M = (Xc * r[:, None]).T @ Xc / n
    Sinv = np.linalg.inv(Sigma)
    H = Sinv @ M @ Sinv
    asym = np.linalg.norm(H - H.T) / np.linalg.norm(H)
    return H, float(np.linalg.cond(Sigma)), float(asym)


# ---------------------------------------------------------------------------
# tables and figures
# ---------------------------------------------------------------------------
def matrix_frame(H: np.ndarray, label: str) -> pd.DataFrame:
    """Return H as a labelled DataFrame (rows/cols 'Load k'); label is the index name."""
    names = [f"Load {k+1}" for k in range(H.shape[0])]
    df = pd.DataFrame(H, index=names, columns=names)
    df.index.name = label
    return df


def symmetry_pairs(H: np.ndarray):
    """Every distinct off-diagonal pair (i<j): H[i,j], H[j,i], signed and absolute gap.

    Input: square matrix.  Outputs: a pairs table and a one-row summary with the
    number of distinct pairs and the mean, median and maximum absolute gap.  For
    d=2 there is exactly one distinct pair; for d=3 there are three.  No
    symmetrisation is performed here.
    """
    d = H.shape[0]
    rows = []
    for i in range(d):
        for j in range(i + 1, d):
            gap = H[i, j] - H[j, i]
            rows.append({"pair (i,j)": f"({i+1},{j+1})", "H[i,j]": H[i, j], "H[j,i]": H[j, i],
                         "signed gap H[i,j]-H[j,i]": gap, "abs gap": abs(gap),
                         "abs gap / max(|H[i,j]|,|H[j,i]|)": abs(gap) / max(abs(H[i, j]), abs(H[j, i]))})
    pairs = pd.DataFrame(rows)
    a = pairs["abs gap"].to_numpy()
    summary = pd.DataFrame([{"distinct pairs": len(a), "mean abs gap": np.mean(a),
                             "median abs gap": np.median(a), "max abs gap": np.max(a)}])
    return pairs, summary


def comparison_table(H_cart: np.ndarray, H_phd: np.ndarray, cart_label: str) -> pd.DataFrame:
    """Entry-by-entry table of two estimates and their difference (no truth implied).

    Inputs: two d x d matrices and a label for the CART column.  Output: one row
    per entry (i,j) with both values, CART - PHD, and |CART - PHD|.
    """
    d = H_cart.shape[0]
    rows = [{"entry": f"({i+1},{j+1})", cart_label: H_cart[i, j], "PHD": H_phd[i, j],
             "CART - PHD": H_cart[i, j] - H_phd[i, j], "|CART - PHD|": abs(H_cart[i, j] - H_phd[i, j])}
            for i in range(d) for j in range(d)]
    return pd.DataFrame(rows)


def heatmap(H: np.ndarray, title: str, ax=None, scale: float = 1e6, absolute: bool = False,
            fmt: str = "{:+.3f}"):
    """Annotated heatmap of a d x d matrix in units of `scale` (default millions).

    Inputs: matrix, title, optional axes, scale divisor, absolute (use a
    one-signed colour map and non-signed labels), number format.  Output: the axes.
    """
    x = H / scale
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    lim = max(np.nanmax(np.abs(x)), 1e-12)
    im = ax.imshow(x, cmap="YlOrRd" if absolute else "RdBu_r",
                   vmin=0 if absolute else -lim, vmax=lim)
    d = H.shape[0]
    for i in range(d):
        for j in range(d):
            ax.text(j, i, (fmt if not absolute else "{:.3f}").format(x[i, j]),
                    ha="center", va="center", fontsize=10)
    names = [f"Load {k+1}" for k in range(d)]
    ax.set_xticks(range(d), names); ax.set_yticks(range(d), names)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"entry / {scale:g}" if scale != 1 else "entry")
    return ax


# ---------------------------------------------------------------------------
# controlled check on the real predictor locations
# ---------------------------------------------------------------------------
def known_quadratic_check(nodes: dict, X: np.ndarray, d: int, seed: int = 0):
    """Apply the Week 45 reconstruction to a quadratic with KNOWN Hessian on the
    same CART partition, two ways, to separate geometry from the within-cell
    distribution of the real predictors.

    (a) exact box means: each node's value is replaced by the exact mean of the
        quadratic over the node's box under the uniform measure (the quantity
        the derivation assumes);
    (b) sample means: each node's value is the mean of the quadratic evaluated at
        the real OPFData rows that fall in the box (no observation noise).
    Inputs: node dict, X (n, d), d, RNG seed for the random symmetric Q.
    (c) uniform resample: each node's value is the mean of the quadratic over
        the SAME NUMBER of points as in (b), but drawn uniformly on the box.
        Comparing (b) with (c) separates the effect of where the real rows sit
        inside the box from the effect of having only a few rows per cell.
    Outputs: dict with Q_true, raw matrices for (a), (b), (c), and their rank
    diagnostics.  The known Q is in the predictors' original units.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q = (A + A.T) / 2
    f = Quadratic(Q, rng.normal(size=d), rng.normal())
    import copy
    exact, sample, unif = (copy.deepcopy(nodes) for _ in range(3))
    fx = f(X)
    for i, n in nodes.items():
        exact[i].value = f.box_mean_exact(n.lo, n.hi)
        mask = np.all((X >= n.lo) & (X <= n.hi), axis=1)
        k = int(mask.sum())
        sample[i].value = float(fx[mask].mean()) if k else np.nan
        if k:
            U = n.lo + rng.random((k, d)) * (n.hi - n.lo)
            unif[i].value = float(f(U).mean())
        else:
            unif[i].value = np.nan
    Qa, ia = hessian_from_tree(exact, d)
    Qb, ib = hessian_from_tree(sample, d)
    Qc, ic = hessian_from_tree(unif, d)
    return {"Q_true": Q, "exact_box_means": Qa, "sample_means": Qb, "uniform_resample": Qc,
            "rank_exact": [int(ia[s]["rank"]) for s in range(d)],
            "rank_sample": [int(ib[s]["rank"]) for s in range(d)],
            "rank_uniform": [int(ic[s]["rank"]) for s in range(d)]}


def within_cell_uniformity(nodes: dict, X: np.ndarray) -> pd.DataFrame:
    """For each internal node, compare the sample centroid of the real rows in the
    box with the box centroid, relative to the box width, per axis.

    Under the uniform-measure assumption these offsets are ~0.  Inputs: node
    dict, X.  Output: table with node, depth, n, and |centroid offset| / width
    per axis; last row gives the mean over nodes.
    """
    rows = []
    for n in nodes.values():
        if n.is_leaf:
            continue
        mask = np.all((X >= n.lo) & (X <= n.hi), axis=1)
        if mask.sum() == 0:
            continue
        off = np.abs(X[mask].mean(0) - (n.lo + n.hi) / 2) / (n.hi - n.lo)
        rows.append({"node": n.id, "depth": n.depth, "n": int(mask.sum()),
                     **{f"|offset|/width axis {k+1}": off[k] for k in range(len(off))}})
    df = pd.DataFrame(rows)
    mean = df.drop(columns=["node", "depth", "n"]).mean()
    mean["node"] = "mean"; mean["depth"] = ""; mean["n"] = ""
    return pd.concat([df, mean.to_frame().T], ignore_index=True)
