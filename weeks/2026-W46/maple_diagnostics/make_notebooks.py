"""Generate and execute the maple_diagnostics notebooks (2D and 3D).

Run from this directory:  python make_notebooks.py
Produces maple_opfdata_hessian_2d.ipynb and maple_opfdata_hessian_3d.ipynb with
executed outputs.  The notebooks themselves only import hessian_tools.

Markdown is written as plain strings (not f-strings) so that LaTeX braces render;
dimension-specific words are substituted with str.replace on named tokens.
"""
import nbformat as nbf
from nbclient import NotebookClient


def fill(text: str, d: int) -> str:
    """Substitute dimension-specific tokens into a markdown string."""
    cols = "1, 2" if d == 2 else "1, 2, 3"
    pairs = "(1,2)" if d == 2 else "(1,2), (1,3), (2,3)"
    npairs = "one distinct off-diagonal pair" if d == 2 else "three distinct off-diagonal pairs"
    row_unknowns = r"$(Q_{11}, Q_{12})$" if d == 2 else r"$(Q_{11}, Q_{12}, Q_{13})$"
    eq = (r"\gamma_B-\gamma_A = Q_{11}\,(z_{B,1}-z_{A,1}) + Q_{12}\,(z_{B,2}-z_{A,2})" if d == 2 else
          r"\gamma_B-\gamma_A = Q_{11}\,(z_{B,1}-z_{A,1}) + Q_{12}\,(z_{B,2}-z_{A,2}) + Q_{13}\,(z_{B,3}-z_{A,3})")
    unknowns = (r"three numbers: the curvatures $Q_{11}$, $Q_{22}$ and the interaction $Q_{12}$" if d == 2 else
                r"six numbers: the curvatures $Q_{11}, Q_{22}, Q_{33}$ and the interactions $Q_{12}, Q_{13}, Q_{23}$")
    return (text.replace("@D@", str(d)).replace("@COLS@", cols).replace("@PAIRS@", pairs)
                .replace("@NPAIRS@", npairs).replace("@ROWUNK@", row_unknowns).replace("@EQ@", eq)
                .replace("@UNKNOWNS@", unknowns).replace("@DP1@", str(d + 1)))


def build(d: int) -> str:
    """Assemble, execute and write the notebook for dimension d; return its filename."""
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(fill(s, d)))
    code = lambda s: C.append(nbf.v4.new_code_cell(s.replace("@D@", str(d))))

    md(r"""# Hessian estimation from regression-tree splits vs. PHD: @D@D OPFData study (Maple)

**Objective.** Estimate how the optimal operating cost of an electricity grid curves jointly with @D@ selected
demand inputs. Reconstruct a Hessian from the splits of an ordinary CART tree with **Algorithm 1 of the
manuscript, implemented line by line**, and compare it with a residual-based Principal Hessian Directions (PHD)
estimate. Then determine, with a controlled experiment on the same tree, where any inconsistency comes from.

**Data.** IEEE-118 OPFData subset, 5,000 records, headerless CSV. Column 0 is the optimal cost (response); columns
@COLS@ are per-unit active demands at @D@ load buses (predictors), labelled Load 1..@D@. Columns 1–99 of the file are
consistent with the 99 load elements of the IEEE-118 case (their means sum to 42.40 p.u. $= 4{,}240$ MW; the case
nominal is 4,242 MW). Physical bus identities per column are not established. Only the first @DP1@ columns are read;
the 2 GB file is not used.

**Scope.** Neither matrix is a true Hessian; the data carry no reference derivative. The checks are: whether every
row of the tree system is identified, whether the raw tree matrix is symmetric, how the two estimates differ, and
what the controlled experiment attributes the inconsistency to. Eigenanalysis and physical interpretation are
outside this study.""")

    code("""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
import hessian_tools as L
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 1e3 else f"{v:,.2f}")
display(L.software_versions())""")

    md(r"""## 1. Data and experimental configuration

All 5,000 observations are retained in their original units; nothing is standardised. Two tree settings are reported
throughout: **A** `max_depth=7, min_samples_leaf=1` (scikit-learn's default leaf size, matching the Cedar baseline)
and **B** `max_depth=7, min_samples_leaf=50`.""")
    code('''def load_and_describe(d):
    """Read the response and the first d predictors; print shapes, summary statistics and correlations.

    Args:
        d: number of predictor columns (2 or 3).

    Returns:
        X (n, d), y (n,), and the CSV path that was read.
    """
    X, y, csv_path = L.load_opf(d)
    print("file:", csv_path)
    print("X shape:", X.shape, "| y shape:", y.shape)
    display(L.describe_inputs(X, y))
    print("predictor correlation matrix:")
    display(L.matrix_frame(np.corrcoef(X.T), "corr"))
    return X, y, csv_path


D = @D@
X, y, csv_path = load_and_describe(D)''')

    md(r"""## 2. The method in plain terms

We have @D@ inputs and one output, the cost. We want @UNKNOWNS@.

**Step 1 — one slope per split.** Take an internal node that splits on Load $s$: a box $\prod_k[\ell_k,u_k]$ with a cut
at $\tau$, left-child node mean $\mu_L$ and right-child node mean $\mu_R$ (the average observed cost of the rows in
each child — nothing is sampled). Then

$$
\gamma \;=\; \frac{\mu_R-\mu_L}{(u_s-\ell_s)/2}
$$

is a slope of cost with respect to Load $s$: the two child boxes' centres are half the box width apart along that
axis, so this is rise over run.

**Step 2 — where that slope is measured.** The slope is not "at the cut". For a quadratic cost it equals the
derivative at one point $z$ in the box: along the split axis $z_s=(\ell_s+\tau+u_s)/3$, the average of the three
breakpoints; along every other axis $z_k=(\ell_k+u_k)/2$, the box middle. Each split now reads: *the Load-$s$ slope at
location $z$ is $\gamma$.*

**Step 3 — two slopes make one equation.** Take two Load-1 splits $A$ and $B$. If the cost is quadratic, the Load-1
slope changes linearly as one moves, so

$$
@EQ@ .
$$

The slope changed because we moved along Load 1 (curvature $Q_{11}$ acts) and along the other axes (interactions act).
Two special cases: if $A$ and $B$ sit at the same height in the other axes, only $Q_{11}$ remains; if they sit at the
same Load-1 position but different heights, only the interaction terms remain — *the Load-1 slope changed as I moved
in Load 2*, which is what an interaction means.

**Step 4 — many equations, solve.** Every ordered pair of distinct Load-1 splits with $z_A\neq z_B$ gives one equation
in @ROWUNK@. Stack them into $V_1 q_1 \approx y_1$ and solve by least squares: row 1 of $\tilde Q$. Repeat for each
axis. The pairs must spread in every direction or the equations cannot separate the unknowns — that is the rank
check, $\operatorname{rank}(V_s)=@D@$, else "not identified".

**Step 5 — the symmetry check.** Row 1 gives $\tilde Q_{12}$ from Load-1 splits; row 2 gives $\tilde Q_{21}$ from
Load-2 splits. A true Hessian has $Q_{12}=Q_{21}$, and the two were computed from completely different splits, so
their gap measures the method's error without any truth. Algorithm 1's last line returns
$\hat Q=\tfrac12(\tilde Q+\tilde Q^\top)$; this notebook inspects $\tilde Q$ *before* that line. In @D@D there are
@NPAIRS@: @PAIRS@.

**Where this can fail.** Step 2 assumes rows fill each box evenly — that is where the $1/3$ and the box middle come
from. If they do not, $z$ is mislocated and the Step-3 equations are built on the wrong coordinates. And each $\mu$ is
a node mean over the rows in a child, so it is noisy; the method differences these means twice.""")

    md(r"""## 3. CART fit and first-derivative extraction

The tree is fitted to the raw data. The table lists every internal node: split axis, extent along it, threshold
$\tau$, the two child node means, the Step-1 slope $\gamma$ and the Step-2 location $z$ (columns `c_k`). The root box
is the observed data range per axis.""")
    code('''def fit_trees(X, y, settings):
    """Fit one CART tree per setting and extract its internal splits.

    Args:
        X, y: predictors and response.
        settings: dict name -> DecisionTreeRegressor keyword arguments.

    Returns:
        nodes: dict name -> node dictionary (boxes, thresholds, child node means).
        splits: dict name -> per-node table (axis, extent, tau, node means, gamma, location z).
    """
    nodes, splits = {}, {}
    for k, kw in settings.items():
        tree = L.fit_cart(X, y, **kw)
        nodes[k] = L.nodes_from_cart(tree, X)
        splits[k] = L.split_table(nodes[k])
        print(f"setting {k}: {kw} -> {len(splits[k])} internal nodes, {int(tree.get_n_leaves())} leaves; "
              + ", ".join(f"{s}: {(splits[k]['split axis']==s).sum()}" for s in sorted(splits[k]['split axis'].unique())))
    return nodes, splits


settings = {"A": dict(max_depth=7, min_samples_leaf=1), "B": dict(max_depth=7, min_samples_leaf=50)}
nodes, splits = fit_trees(X, y, settings)
print("\\nsetting B, first rows of the split table:")
display(splits["B"].head(12))''')

    md(r"""## 4. Algorithm 1: $\tilde Q$ before symmetrisation

`hessian_tools.algorithm1_hessian` follows the pseudocode line by line (lines 1–4 slopes and locations; 5–11 ordered
pairs with $s_A=s_B$, $z_A\neq z_B$; 12–18 rank check and least squares per row; 19 symmetrisation). Diagnostics per
row $s$: nodes splitting axis $s$, rows in $V_s$ (ordered pairs, so twice the number of unordered pairs), rank of
$V_s$, condition number.

Two cross-checks: the Week 45 solver (unordered pairs, same equations) and the Cedar centred-regression formulation
must both reproduce $\tilde Q$ to rounding.""")
    code('''def run_algorithm1(nodes, d):
    """Run Algorithm 1 on every tree setting and cross-check it against two other formulations.

    Args:
        nodes: dict name -> node dictionary from fit_trees.
        d: dimension.

    Returns:
        Q_tilde: dict name -> matrix before line 19 (not symmetrised).
        Q_hat: dict name -> symmetrised output of line 19.
        diag: dict name -> per-row rank/condition table.
    """
    Q_tilde, Q_hat, diag = {}, {}, {}
    for k in nodes:
        res = L.algorithm1_hessian(nodes[k], d)
        Q_tilde[k], Q_hat[k], diag[k] = res["Q_tilde"], res["Q_hat"], res["diag"]
        w45, _ = L.cart_hessian_raw(nodes[k], d)
        alt = L.centred_regression_rows(nodes[k], d)
        print(f"--- setting {k}: identified = {res['identified']} ---")
        display(diag[k])
        print("Q_tilde (before line 19, NOT symmetrised):")
        display(L.matrix_frame(Q_tilde[k], f"Q_tilde {k}"))
        print(f"max |Algorithm 1 - Week 45 solver| = {np.nanmax(np.abs(Q_tilde[k]-w45)):.2e} | "
              f"max |Algorithm 1 - centred regression| = {np.nanmax(np.abs(Q_tilde[k]-alt)):.2e}\\n")
    return Q_tilde, Q_hat, diag


Q_tilde, Q_hat, diag = run_algorithm1(nodes, D)''')

    md(r"""## 5. PHD: curvature from residuals

Residual-based Principal Hessian Directions (Li, 1992). Centre the inputs, $X_c = X-\bar x$; covariance
$\Sigma = X_c^\top X_c/n$ (divisor $n$); fit an ordinary least-squares line $y \approx \beta_0+\beta^\top x$ and keep the
residuals $r$; form $M=\frac1n\sum_i r_i\,x_{c,i}x_{c,i}^\top$; estimate

$$
\hat H_{\mathrm{PHD}} \;=\; \Sigma^{-1} M\, \Sigma^{-1},
$$

in original units (cost per squared demand unit). Under a Gaussian input distribution $M$ identifies the average
Hessian of $E[y\mid x]$; the loads here are products of uniform factors and correlated, so PHD is a baseline with an
assumption, not a reference. $M$ is symmetric by construction, so the reported asymmetry of $\hat H_{\mathrm{PHD}}$
is at rounding level.""")
    code('''def run_phd(X, y):
    """Compute the residual-based PHD estimate and print its diagnostics.

    Args:
        X, y: predictors and response in original units.

    Returns:
        H_phd: d by d matrix, Sigma^{-1} M Sigma^{-1}.
    """
    H_phd, cond_sigma, asym_phd = L.phd_residual(X, y)
    print(f"condition number of Sigma: {cond_sigma:.2f} | Frobenius asymmetry of H_PHD: {asym_phd:.1e}")
    display(L.matrix_frame(H_phd, "PHD"))
    return H_phd


H_phd = run_phd(X, y)''')

    md(r"""## 6. Estimates side by side (original units)""")
    code('''def show_estimates(Q_tilde, Q_hat, H_phd):
    """Print every estimate in original units: Q_tilde and Q_hat per setting, then PHD."""
    for k in Q_tilde:
        print(f"Algorithm 1, setting {k} — Q_tilde (before line 19):"); display(L.matrix_frame(Q_tilde[k], f"Q_tilde {k}"))
        print(f"Algorithm 1, setting {k} — Q_hat (line 19 output):"); display(L.matrix_frame(Q_hat[k], f"Q_hat {k}"))
    print("PHD:"); display(L.matrix_frame(H_phd, "PHD"))


show_estimates(Q_tilde, Q_hat, H_phd)''')

    md(r"""## 7. Numerical symmetry check of $\tilde Q$

For each distinct pair $i<j$: $\tilde Q_{ij}$ (from the row solved with Load-$i$ splits), $\tilde Q_{ji}$ (from Load-$j$
splits), the signed gap, the absolute gap, and the gap relative to the larger of the two entries. The mean, median and
maximum are summaries across the @NPAIRS@ — @PAIRS@ — not across observations. Nothing is symmetrised here.""")
    code('''def report_symmetry(Q_tilde):
    """Show every distinct off-diagonal pair of Q_tilde and the mean/median/max absolute gap, per setting.

    Args:
        Q_tilde: dict name -> matrix before line 19.

    Returns:
        dict name -> (pairs table, summary table).
    """
    out = {}
    for k in Q_tilde:
        pairs, summary = L.symmetry_pairs(Q_tilde[k])
        print(f"--- setting {k} ---"); display(pairs); display(summary)
        out[k] = (pairs, summary)
    return out


symmetry = report_symmetry(Q_tilde)''')

    md(r"""## 8. Disagreement with PHD

First $\tilde Q$ against PHD entry by entry; then $\hat Q=\tfrac12(\tilde Q+\tilde Q^\top)$ against PHD, reported only
after the symmetry table so that line 19 is a visible step. Differences are between two estimates; no column is a
truth value.""")
    code('''def compare_with_phd(Q_tilde, Q_hat, H_phd):
    """Entry-by-entry disagreement tables: Q_tilde vs PHD, then Q_hat vs PHD, per setting.

    Args:
        Q_tilde, Q_hat: dicts name -> matrix.
        H_phd: PHD matrix.
    """
    for k in Q_tilde:
        print(f"--- setting {k}: Q_tilde vs PHD ---")
        display(L.comparison_table(Q_tilde[k], H_phd, f"Q_tilde {k}"))
        asym = np.linalg.norm(Q_tilde[k] - Q_tilde[k].T) / np.linalg.norm(Q_tilde[k])
        print(f"--- setting {k}: Q_hat vs PHD   (asymmetry of Q_tilde ||Q-Q^T||/||Q|| = {asym:.3f}) ---")
        display(L.comparison_table(Q_hat[k], H_phd, f"Q_hat {k}"))


compare_with_phd(Q_tilde, Q_hat, H_phd)''')

    md(r"""## 9. Figures

Each matrix is shown directly. Colour indicates sign and magnitude; annotations are in **millions of cost units per
squared demand unit**. The $\tilde Q$ figures are intentionally not symmetrised.""")
    code('''def plot_hessian(matrix, title, absolute=False):
    """Display one annotated matrix in millions of original units.

    Args:
        matrix: square matrix to plot.
        title: figure title naming the estimator or diagnostic.
        absolute: use a non-negative colour scale (for gaps and |differences|).
    """
    values = np.asarray(matrix, float) / 1e6
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    if absolute:
        image = ax.imshow(values, cmap="YlOrRd", vmin=0)
    else:
        vmax = max(0.1, float(np.nanmax(np.abs(values))))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > 0.60 * max(np.nanmax(np.abs(values)), 1e-12) else "black"
            ax.text(j, i, f"{values[i, j]:+.3f}" if not absolute else f"{values[i, j]:.3f}",
                    ha="center", va="center", fontsize=12, weight="medium", color=color)
    ticks = np.arange(values.shape[0])
    ax.set_xticks(ticks, [f"Load {j+1}" for j in ticks]); ax.set_yticks(ticks, [f"Load {j+1}" for j in ticks])
    ax.set_xlabel("Demand variable (column)"); ax.set_ylabel("Demand variable (row)")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    fig.colorbar(image, ax=ax, shrink=.83, label="Matrix entries (millions)")
    fig.tight_layout(); plt.show()


def plot_all(Q_tilde, Q_hat, H_phd):
    """Plot PHD, then for each setting: Q_tilde, its symmetry discrepancy, and |Q_hat - PHD|."""
    plot_hessian(H_phd, "PHD estimated Hessian")
    for k in Q_tilde:
        plot_hessian(Q_tilde[k], f"Algorithm 1, setting {k}: Q_tilde (raw rows)")
        plot_hessian(np.abs(Q_tilde[k] - Q_tilde[k].T), f"Setting {k}: symmetry discrepancy |Q_tilde - Q_tilde^T|", absolute=True)
        plot_hessian(np.abs(Q_hat[k] - H_phd), f"Setting {k}: |Q_hat - PHD| (disagreement, not error)", absolute=True)


plot_all(Q_tilde, Q_hat, H_phd)''')

    md(r"""## 10. Controlled check on the same partitions: where does the inconsistency come from?

Algorithm 1 is exact for a quadratic when each node's value is the exact mean of that quadratic over the node's box
under the uniform measure. On the fitted partitions above, a quadratic with a known random symmetric Hessian $Q$ is
evaluated three ways and reconstructed with the same code:

- **(a) exact box means** — the integral of the quadratic over each box: tests the geometry and the code only;
- **(b) node means at the real rows** — the quadratic evaluated at the OPFData rows in each box and averaged, no
  observation noise: adds the effect of how many rows each box holds *and* where they sit;
- **(c) uniform resample** — the same number of points per box as in (b), placed uniformly at random on the box
  (the one place in this notebook where anything is sampled): adds only the "how many rows" effect.

Errors are $\max|\tilde Q-Q|/\max|Q|$. The final table reports, per internal node, how far the real rows' centroid
sits from the box centre relative to the box width (0 under the uniform assumption).""")
    code('''def controlled_check(nodes, X, d, seeds=(0, 1, 2)):
    """Run the known-quadratic check (a)/(b)/(c) on every setting and seed.

    Args:
        nodes: dict name -> node dictionary.
        X: predictors (used for the real-row node means and the per-box counts).
        d: dimension.
        seeds: random seeds for the known Hessian.

    Returns:
        DataFrame, one row per (setting, seed): relative max error of each version and the ranks.
    """
    rows = []
    for k in nodes:
        for seed in seeds:
            c = L.known_quadratic_check(nodes[k], X, d, seed)
            scale = np.max(np.abs(c["Q_true"]))
            rows.append({"setting": k, "seed": seed,
                         "(a) exact box means": np.nanmax(np.abs(c["exact_box_means"] - c["Q_true"])) / scale,
                         "(b) node means, real rows": np.nanmax(np.abs(c["sample_means"] - c["Q_true"])) / scale,
                         "(c) uniform resample": np.nanmax(np.abs(c["uniform_resample"] - c["Q_true"])) / scale,
                         "ranks (a/b/c)": f"{c['rank_exact']}/{c['rank_sample']}/{c['rank_uniform']}"})
    return pd.DataFrame(rows)


def show_one_check(nodes, X, d, setting="B", seed=0):
    """Print the known Q and its three reconstructions for one setting and seed."""
    c = L.known_quadratic_check(nodes[setting], X, d, seed)
    print(f"setting {setting}, seed {seed} — known Q:"); display(L.matrix_frame(c["Q_true"], "Q true"))
    print("(a) exact box means:"); display(L.matrix_frame(c["exact_box_means"], "(a)"))
    print("(b) node means at the real rows:"); display(L.matrix_frame(c["sample_means"], "(b)"))
    print("(c) uniform resample:"); display(L.matrix_frame(c["uniform_resample"], "(c)"))


ctrl = controlled_check(nodes, X, D)
with pd.option_context("display.float_format", lambda v: f"{v:.2e}" if v < 1e-6 else f"{v:.2f}"):
    display(ctrl)
show_one_check(nodes, X, D)''')
    code('''def show_within_box_offsets(nodes, X, setting="B", tail=6):
    """Print the last rows of the per-node centroid-offset table (last row = mean over nodes)."""
    u = L.within_cell_uniformity(nodes[setting], X)
    print(f"setting {setting}: |row centroid − box centroid| / box width per axis; last row = mean over internal nodes")
    display(u.tail(tail))


show_within_box_offsets(nodes, X)''')

    md(r"""## 11. Findings and limitations

1. **The implementation is Algorithm 1.** The Week 45 solver and the Cedar centred regression reproduce $\tilde Q$ to
   rounding; rank is full in every row; and with exact box means the known quadratic is recovered to $\sim 10^{-14}$
   (§10a). There is no coding defect in the reconstruction.
2. **$\tilde Q$ is not symmetric on this data**, by amounts comparable to or larger than the entries (§7). Because
   row $i$ and row $j$ come from disjoint node sets, $\tilde Q_{ij}$ and $\tilde Q_{ji}$ are independent estimates of
   one quantity; their gap is a direct, assumption-free measure of estimation error. Line 19 averages it away, so the
   published output $\hat Q$ cannot show it.
3. **Source of the inconsistency.** Replacing exact box means by node means of a *noise-free* known quadratic already
   produces errors of order 1–10 times the true entries (§10b). A uniform resample with matched counts (§10c) gives
   errors of the same order, so the dominant cause at $n=5{,}000$ is node-mean variability in small boxes — second
   differences of noisy first differences. At setting B the real rows (b) are worse than the uniform resample (c) in
   every seed, so the non-uniform placement of rows inside boxes (centroid offsets of roughly 7–11 % of box width)
   adds a further error; at setting A the two effects are not separable. Larger leaves reduce the inconsistency but
   do not remove it.
4. **Against PHD.** PHD is symmetric with entries of order $10^5$–$10^6$ and disagrees with the tree estimates in
   magnitude and, for some entries, in sign (§8). Given (2)–(3) this disagreement is within the tree estimate's own
   inconsistency and does not yet discriminate between the methods' assumptions.

**Limitations.** 5,000 rows at depth 7 is far below what second differences of node means require; the full dataset
was deliberately not used. The Step-2 location assumes a uniform within-box distribution that the correlated loads
do not satisfy (see `../diagnostics/predictor_uniformity`). All node pairs are weighted equally and pooled over the
whole domain (global quadratic assumption). PHD assumes a Gaussian design. Physical column identities beyond
"cost" and "per-unit demand" are not verified. No eigenanalysis, bootstrap, prediction metric or alternative
estimator is included.""")

    nb["cells"] = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    fname = f"maple_opfdata_hessian_{d}d.ipynb"
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbf.write(nb, fname)
    return fname


if __name__ == "__main__":
    for d in (2, 3):
        print("wrote", build(d))
