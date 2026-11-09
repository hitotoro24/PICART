"""PHD and tree-derived Hessian comparison for a small OPFData subset.

Assumptions: headerless CSV, response in column 0, selected demand columns 1..d.
Neither resulting matrix is treated as a known true Hessian.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor

DEFAULT_CSV = Path('/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv')


@dataclass
class Split:
    """One CART split: feature, box bounds, threshold, and observed child means."""
    s: int
    lo: np.ndarray
    hi: np.ndarray
    tau: float
    left_mean: float
    right_mean: float


def read_inputs(d: int, path: Path = DEFAULT_CSV):
    """Read 2 or 3 selected demand features and cost; return X, y, and file path."""
    if d not in (2, 3):
        raise ValueError('This first study is limited to d=2 or d=3.')
    actual = next((p for p in (path, Path('118_ieee_smol.csv'), Path('/mnt/data/118_ieee_smol.csv')) if p.is_file()), None)
    if actual is None:
        raise FileNotFoundError(f'OPFData CSV not found at {path}')
    data = pd.read_csv(actual, header=None, usecols=range(d+1)).to_numpy(float)
    if not np.isfinite(data).all():
        raise ValueError('Selected columns contain nonfinite values')
    y, X = data[:, 0], data[:, 1:]
    return X, y, actual


def phd_residual(X: np.ndarray, y: np.ndarray):
    """Return residual-based PHD matrix and covariance condition number, in original units."""
    z = X - X.mean(axis=0)
    design = np.column_stack([np.ones(len(y)), z])
    r = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    covariance = z.T @ z / len(y)
    condition = np.linalg.cond(covariance)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError('Predictor covariance is too ill-conditioned')
    weighted = z.T @ (r[:, None] * z) / len(y)
    a = np.linalg.solve(covariance, weighted)
    h = np.linalg.solve(covariance, a.T).T
    return (h + h.T) / 2, float(condition)


def extract_splits(tree, lower: np.ndarray, upper: np.ndarray):
    """Read split geometry and child means from sklearn CART; return a list of splits."""
    t = tree.tree_
    stack = [(0, lower.copy(), upper.copy())]
    rows = []
    while stack:
        node, lo, hi = stack.pop()
        left = int(t.children_left[node])
        if left < 0:
            continue
        right = int(t.children_right[node]); s = int(t.feature[node]); tau = float(t.threshold[node])
        rows.append(Split(s, lo.copy(), hi.copy(), tau,
                          float(t.value[left].ravel()[0]), float(t.value[right].ravel()[0])))
        left_hi = hi.copy(); left_hi[s] = tau
        right_lo = lo.copy(); right_lo[s] = tau
        stack.extend([(left, lo.copy(), left_hi), (right, right_lo, hi.copy())])
    return rows


def cart_hessian(splits: list[Split], d: int):
    """Return raw CART Hessian and per-row rank/condition diagnostics from effective centers."""
    h = np.full((d, d), np.nan)
    checks = []
    for s in range(d):
        nodes = [v for v in splits if v.s == s]
        if len(nodes) < d + 1:
            checks.append((s + 1, len(nodes), 0, np.inf))
            continue
        centers = np.array([(v.lo + v.hi)/2 for v in nodes])
        gradient = np.array([2 * (v.right_mean - v.left_mean)/(v.hi[s]-v.lo[s]) for v in nodes])
        for i, v in enumerate(nodes):
            centers[i, s] = (v.lo[s] + v.tau + v.hi[s])/3
        v = centers - centers.mean(axis=0)
        g = gradient - gradient.mean()
        solution, _, rank, singular_values = np.linalg.lstsq(v, g, rcond=1e-10)
        condition = singular_values[0]/singular_values[-1] if rank == d else np.inf
        if rank == d:
            h[s] = solution
        checks.append((s+1, len(nodes), int(rank), float(condition)))
    return h, pd.DataFrame(checks, columns=['Load', 'Split nodes', 'Rank', 'Condition'])


def symmetry_table(h: np.ndarray):
    """Return pairwise raw off-diagonal values and mean/median/max absolute gaps."""
    if not np.isfinite(h).all():
        raise ValueError('Some Hessian rows are unidentified')
    rows=[]
    for i in range(len(h)):
        for j in range(i+1, len(h)):
            gap=h[i,j]-h[j,i]
            rows.append([f'({i+1}, {j+1})', h[i,j], h[j,i], gap, abs(gap)])
    pairs=pd.DataFrame(rows, columns=['Pair','H[i,j]','H[j,i]','Signed gap','Absolute gap'])
    a=pairs['Absolute gap'].to_numpy()
    stats=pd.DataFrame([{'Pairs':len(a),'Mean abs gap':np.mean(a),'Median abs gap':np.median(a),'Max abs gap':np.max(a)}])
    return pairs,stats


def show_matrix(h: np.ndarray, title: str, absolute: bool = False):
    """Display an annotated heatmap of one matrix, in millions of original units."""
    x = h/1e6
    fig, ax = plt.subplots(figsize=(5, 4))
    limit=max(0.1, np.max(abs(x)))
    im=ax.imshow(x, cmap='YlOrRd' if absolute else 'RdBu_r',
                 vmin=0 if absolute else -limit, vmax=limit)
    for i in range(len(x)):
        for j in range(len(x)):
            ax.text(j,i,f'{x[i,j]:+.3f}' if not absolute else f'{x[i,j]:.3f}',
                    ha='center',va='center',fontsize=11)
    ax.set_xticks(range(len(x)), [f'Load {i+1}' for i in range(len(x))])
    ax.set_yticks(range(len(x)), [f'Load {i+1}' for i in range(len(x))])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Matrix entry (millions)')
    fig.tight_layout(); plt.show()


def run(d: int):
    """Run the PHD–CART comparison on d selected demand columns and display results."""
    X,y,path=read_inputs(d)
    phd,cond=phd_residual(X,y)
    tree=DecisionTreeRegressor(max_depth=7,random_state=0).fit(X,y)
    raw,diagnostics=cart_hessian(extract_splits(tree,X.min(0),X.max(0)),d)
    print(f'{d}D | n={len(y):,} | CSV={path} | PHD covariance condition={cond:.2f}')
    print('PHD matrix:\n',phd)
    print('Raw CART matrix (not symmetrized):\n',raw)
    display(diagnostics)
    if not np.isfinite(raw).all():
        print('Not all CART rows identifiable; comparison stopped.')
        return
    pairs,stats=symmetry_table(raw)
    display(pairs.style.format({c:'{:,.2f}' for c in ['H[i,j]','H[j,i]','Signed gap','Absolute gap']}))
    display(stats.style.format({c:'{:,.2f}' for c in ['Mean abs gap','Median abs gap','Max abs gap']}))
    print(f'Maximum absolute difference, raw CART versus PHD: {np.max(abs(raw-phd)):,.2f} (not error to truth)')
    show_matrix(phd,'Residual-based PHD')
    show_matrix(raw,'CART Hessian (raw)')
    show_matrix(abs(raw-raw.T),'Absolute symmetry discrepancy',absolute=True)
    return {'phd':phd,'cart_raw':raw,'symmetry_pairs':pairs,'symmetry_summary':stats}
