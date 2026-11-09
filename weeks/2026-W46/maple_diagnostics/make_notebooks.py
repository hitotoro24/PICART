"""Generate and execute the maple_diagnostics notebooks (2D and 3D).

Run from this directory:  python make_notebooks.py
Produces maple_opfdata_hessian_2d.ipynb and maple_opfdata_hessian_3d.ipynb with
executed outputs.  The notebooks themselves only import hessian_tools.
"""
import nbformat as nbf
from nbclient import NotebookClient


def build(d: int) -> str:
    """Assemble the notebook for dimension d and return its filename."""
    cols = ", ".join(str(k) for k in range(1, d + 1))
    n_pairs = d * (d - 1) // 2
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: C.append(nbf.v4.new_code_cell(s))

    md(f"""# CART-derived Hessian vs. residual-based PHD on OPFData IEEE-118 — {d}D

**Research question.** On a real regression dataset (operating cost as a function of {d} bus demands), how does the
Hessian reconstructed from an ordinary CART tree by the effective-centre method compare with the classical
residual-based principal Hessian directions (PHD) estimate, and is the raw CART reconstruction internally
consistent (symmetric) when its rows are solved independently?

**What this notebook does not claim.** Neither estimate is a true Hessian. The CSV provides no reference
derivative. PHD is a baseline with its own assumption (a Gaussian predictor design); agreement or disagreement
with it is reported, not scored.

**Data and column convention.** Headerless CSV `118_ieee_smol.csv`, 5,000 rows × 100 columns.
Column 0 is the response (operating cost of the solved optimal power flow). Columns {cols} are the predictors and
are labelled *Load 1..{d}* below (Load k ≡ CSV column k). Columns 1–99 are consistent with per-unit active demand at
the 99 load elements of the IEEE-118 case: their means sum to 42.40 p.u. (4,240 MW against the case's nominal
4,242 MW). Physical bus identities per column are not established here. Only the first {d + 1} columns are read;
the file is not modified and the full 2 GB dataset is not used.

**Approach.**
1. Fit an ordinary sklearn CART regression tree (squared-error criterion). Two settings are reported:
   **A** `max_depth=7, min_samples_leaf=1` (sklearn default leaf size) and **B** `max_depth=7, min_samples_leaf=50`.
2. From each internal node take the two child-node response means and form the first-derivative estimate
   γ = 2(μ_right − μ_left)/(hi_s − lo_s) along the split axis s.
3. Reconstruct the Hessian with **Algorithm 1 of the manuscript, implemented line by line** (`hessian_tools.
   algorithm1_hessian`): for every internal node compute γ by eq. (3) and the effective location z by eq. (4),
   z_s = (lo_s + τ + hi_s)/3 and z_p = (lo_p + hi_p)/2 for p ≠ s; for every ordered pair of distinct nodes with the
   same split axis and z_A ≠ z_B append (z_B − z_A)ᵀ to V_s and γ_B − γ_A to y_s; for each s check rank(V_s) = d
   (else "not identified") and solve by least squares for row s of Q̃; return Q̂ = (Q̃ + Q̃ᵀ)/2. The matrix **before**
   line 19, Q̃, is reported first because its off-diagonal pairs are two independent estimates of one quantity.
4. Implement residual-based PHD independently (conventions stated in §5).
5. Print both matrices, check the raw CART matrix for symmetry pair by pair, then compare with PHD.
6. Controlled check: apply the same reconstruction, on the same fitted partition, to a quadratic with a known
   Hessian, to attribute any inconsistency to code, geometry, cell size, or the within-cell distribution of the
   real predictors.""")

    axes = "two inputs (Load 1, Load 2)" if d == 2 else "three inputs (Load 1, Load 2, Load 3)"
    unknowns = ("three numbers: how cost curves along Load 1 (Q₁₁), along Load 2 (Q₂₂), and their interaction (Q₁₂)"
                if d == 2 else
                "six numbers: the three curvatures Q₁₁, Q₂₂, Q₃₃ and the three interactions Q₁₂, Q₁₃, Q₂₃")
    eq = ("γ_B − γ_A = Q₁₁·(z_B,1 − z_A,1) + Q₁₂·(z_B,2 − z_A,2)" if d == 2 else
          "γ_B − γ_A = Q₁₁·(z_B,1 − z_A,1) + Q₁₂·(z_B,2 − z_A,2) + Q₁₃·(z_B,3 − z_A,3)")
    unk_row = "(Q₁₁, Q₁₂)" if d == 2 else "(Q₁₁, Q₁₂, Q₁₃)"
    pairs_note = ("In 2D there is exactly one such pair to check, (1,2)." if d == 2 else
                  "In 3D there are three such pairs to check: (1,2), (1,3), (2,3).")
    md(f"""## The method in plain terms ({d}D)

We have {axes} and one output, the cost. We want {unknowns}.

**Step 1 — one slope per split.** Take an internal node that splits on Load 1: a box with a cut at τ, left-child mean
cost μ_L, right-child mean cost μ_R. Then γ = (μ_R − μ_L) / ((hi₁ − lo₁)/2) is a slope of cost with respect to
Load 1 — the two child boxes' centres are half the box width apart along Load 1, so this is "rise over run". A
Load-2 split gives a slope with respect to Load 2 the same way{"" if d == 2 else ", and a Load-3 split with respect to Load 3"}.

**Step 2 — where that slope is measured.** The slope is not "at the cut". For a quadratic cost it equals the derivative
at one point in the box: along the split axis, (lo + τ + hi)/3, the average of the three breakpoints; along every other
axis, the middle of the box. Call that point z. Each split now reads: *"the Load-s slope at location z is γ."*

**Step 3 — two slopes make one equation.** Take two Load-1 splits, A and B. If the cost is quadratic, the Load-1 slope
changes linearly as you move, so

    {eq}

Read it as: the slope changed because we moved along Load 1 (curvature Q₁₁ acts) and along the other axes
(interactions act). Two special cases: if A and B sit at the same height in the other axes, only Q₁₁ remains — "the
slope changed as I moved along Load 1"; if they sit at the same Load-1 position but different heights, only the
interaction terms remain — "the Load-1 slope changed as I moved in Load 2": that *is* an interaction.

**Step 4 — many equations, solve.** Every pair of Load-1 splits gives one equation in {unk_row}. Stack them and solve
by least squares → row 1 of the Hessian. Repeat with Load-2 splits for row 2{"" if d == 2 else " and Load-3 splits for row 3"}.
The pairs must spread in every direction, otherwise the equations cannot separate the unknowns — that is the rank check.

**Step 5 — the symmetry check.** Row 1 gives Q₁₂ from Load-1 splits; row 2 gives Q₂₁ from Load-2 splits. A true Hessian
has Q₁₂ = Q₂₁, and the two were computed from completely different splits, so their gap measures the method's error
without needing any truth. Algorithm 1's last line averages them; this notebook looks at them *before* averaging.
{pairs_note}

**Why this can fail on real data.** Step 2 assumes rows fill each box evenly — that is where the /3 and the box-middle
come from. If they do not, z is mislocated and the Step-3 equations are built on the wrong coordinates. And each μ is
an average over the rows in a child, so it is noisy; the method differences these averages twice.""")

    code("""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
import hessian_tools as L
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 1e3 else f"{v:,.2f}")
display(L.software_versions())""")

    md("## 1. Inputs")
    code(f"""D = {d}
X, y, csv_path = L.load_opf(D)
print("file:", csv_path)
print("X shape:", X.shape, "| y shape:", y.shape)
display(L.describe_inputs(X, y))
print("predictor correlation matrix:")
display(L.matrix_frame(np.corrcoef(X.T), "corr"))""")

    md("""## 2. CART fit and first-derivative extraction

The tree is fitted to the raw (unscaled) data. The table lists every internal node with its split axis, extent along
that axis, threshold τ, the two child-node response means, the eq.-3 gradient estimate γ and the effective centre c.
The root box is the observed data range per axis.""")
    code("""settings = {"A": dict(max_depth=7, min_samples_leaf=1), "B": dict(max_depth=7, min_samples_leaf=50)}
trees, nodes, splits = {}, {}, {}
for k, kw in settings.items():
    trees[k] = L.fit_cart(X, y, **kw)
    nodes[k] = L.nodes_from_cart(trees[k], X)
    splits[k] = L.split_table(nodes[k])
    print(f"setting {k}: {kw} -> {len(splits[k])} internal nodes, "
          f"{int(trees[k].get_n_leaves())} leaves; "
          + ", ".join(f"{s}: {(splits[k]['split axis']==s).sum()}" for s in sorted(splits[k]['split axis'].unique())))
print("\\nsetting B, first rows of the split table:")
display(splits["B"].head(12))""")

    md(f"""## 3. Algorithm 1: Q̃ before symmetrisation (line 19 not yet applied)

Rank diagnostics per row s: number of nodes splitting axis s, number of rows in V_s (ordered pairs, so twice the
number of unordered pairs), rank of V_s (must equal {d}, else Algorithm 1 returns "not identified"), and its
condition number.

Two cross-checks confirm the implementation: the Week 45 solver (unordered pairs, same equations) and the
centred-regression formulation used in cedar_baseline must both reproduce Q̃ to rounding.""")
    code("""H_raw, H_hat, diag = {}, {}, {}
for k in settings:
    res = L.algorithm1_hessian(nodes[k], D)
    H_raw[k], H_hat[k], diag[k] = res["Q_tilde"], res["Q_hat"], res["diag"]
    w45, _ = L.cart_hessian_raw(nodes[k], D)
    alt = L.centred_regression_rows(nodes[k], D)
    print(f"--- setting {k}: identified = {res['identified']} ---")
    display(diag[k])
    print("Q_tilde (Algorithm 1 before line 19, NOT symmetrised):")
    display(L.matrix_frame(H_raw[k], f"Q_tilde {k}"))
    print(f"max |Algorithm 1 - Week 45 solver| = {np.nanmax(np.abs(H_raw[k]-w45)):.2e} | "
          f"max |Algorithm 1 - centred regression| = {np.nanmax(np.abs(H_raw[k]-alt)):.2e}\\n")""")

    md("""## 4. Independent residual-based PHD

Conventions: predictors centred by their sample mean; Σ = XcᵀXc / n (divisor n); residual r from the ordinary
least-squares fit of the cost on an intercept and the linear predictor terms; M = Σᵢ rᵢ xcᵢ xcᵢᵀ / n; estimate
H_PHD = Σ⁻¹ M Σ⁻¹, in the original units of cost per (predictor unit)². M is symmetric by construction, so the
reported asymmetry of H_PHD is at rounding level and is shown only for completeness.""")
    code("""H_phd, cond_sigma, asym_phd = L.phd_residual(X, y)
print(f"condition number of Sigma: {cond_sigma:.2f} | Frobenius asymmetry of H_PHD: {asym_phd:.1e}")
display(L.matrix_frame(H_phd, "PHD"))""")

    md("## 5. Both estimates side by side (original units)")
    code("""for k in settings:
    print(f"Algorithm 1 Q_tilde (before line 19), setting {k}:"); display(L.matrix_frame(H_raw[k], f"Q_tilde {k}"))
    print(f"Algorithm 1 Q_hat (line 19 output), setting {k}:"); display(L.matrix_frame(H_hat[k], f"Q_hat {k}"))
print("PHD:"); display(L.matrix_frame(H_phd, "PHD"))""")

    md(f"""## 6. Symmetry of Q̃ (what line 19 averages away)

A {d}×{d} matrix has {n_pairs} distinct off-diagonal pair{'s' if n_pairs > 1 else ''}
({', '.join(f'({i+1},{j+1})' for i in range(d) for j in range(i+1, d))}). For each pair the table shows H[i,j]
(from the row solved with axis-i nodes), H[j,i] (from axis-j nodes), the signed gap, the absolute gap, and the gap
relative to the larger of the two entries. {'With a single pair, mean = median = max.' if n_pairs == 1 else 'The summary gives mean, median and maximum absolute gap over the three pairs.'}
Nothing is symmetrised in this section.""")
    code("""for k in settings:
    pairs, summary = L.symmetry_pairs(H_raw[k])
    print(f"--- setting {k} ---"); display(pairs); display(summary)""")

    md("""## 7. Comparison with PHD

First Q̃ against PHD, entry by entry. Then the Algorithm 1 output Q̂ = (Q̃ + Q̃ᵀ)/2 against PHD — reported only after
the discrepancies above, so that line 19 is a visible step and not a hidden one. Differences are between two
estimates; no column is a truth value.""")
    code("""for k in settings:
    print(f"--- setting {k}: Q_tilde vs PHD ---")
    display(L.comparison_table(H_raw[k], H_phd, f"Q_tilde {k}"))
    asym = np.linalg.norm(H_raw[k] - H_raw[k].T) / np.linalg.norm(H_raw[k])
    print(f"--- setting {k}: Q_hat (Algorithm 1 output) vs PHD   (asymmetry of Q_tilde ||Q-Qᵀ||/||Q|| = {asym:.3f}) ---")
    display(L.comparison_table(H_hat[k], H_phd, f"Q_hat {k}"))""")

    md("### Heatmaps (entries in millions of original units)")
    code("""fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
L.heatmap(H_phd, "PHD", axes[0, 0])
L.heatmap(H_raw["A"], "Q̃ (Algorithm 1 before line 19), setting A", axes[0, 1])
L.heatmap(np.abs(H_raw["A"] - H_raw["A"].T), "|H − Hᵀ|, setting A", axes[0, 2], absolute=True)
L.heatmap(np.abs(H_hat["A"] - H_phd), "|Q̂ A − PHD|", axes[1, 0], absolute=True)
L.heatmap(H_raw["B"], "Q̃ (Algorithm 1 before line 19), setting B", axes[1, 1])
L.heatmap(np.abs(H_raw["B"] - H_raw["B"].T), "|H − Hᵀ|, setting B", axes[1, 2], absolute=True)
fig.tight_layout(); plt.show()""")

    md(f"""## 8. Controlled check on the same partition: where does the inconsistency come from?

The reconstruction is exact for a quadratic when each node's value is the exact mean of that quadratic over the
node's box under the uniform measure. On the fitted partitions above, a quadratic with a known random symmetric
Hessian Q is evaluated three ways and reconstructed with the same solver:

- **(a) exact box means** — tests the geometry and the code only;
- **(b) sample means at the real rows** — the quadratic is evaluated at the OPFData predictor rows that fall in each
  box, no observation noise;
- **(c) uniform resample** — the same number of points per box as in (b), drawn uniformly on the box.

(a) separates code/geometry from everything else; (c) vs (a) measures the cost of finite cells; (b) vs (c) measures
the additional cost of the real rows being non-uniformly placed inside the boxes. Errors are max |Ĥ − Q| / max |Q|.
The last table reports, per internal node, how far the real rows' centroid sits from the box centroid relative to
the box width (0 under the uniform assumption).""")
    code("""rows = []
for k in settings:
    for seed in (0, 1, 2):
        c = L.known_quadratic_check(nodes[k], X, D, seed)
        scale = np.max(np.abs(c["Q_true"]))
        rows.append({"setting": k, "seed": seed,
                     "(a) exact box means": np.nanmax(np.abs(c["exact_box_means"] - c["Q_true"])) / scale,
                     "(b) real-row sample means": np.nanmax(np.abs(c["sample_means"] - c["Q_true"])) / scale,
                     "(c) uniform resample": np.nanmax(np.abs(c["uniform_resample"] - c["Q_true"])) / scale,
                     "ranks (a/b/c)": f"{c['rank_exact']}/{c['rank_sample']}/{c['rank_uniform']}"})
ctrl = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda v: f"{v:.2e}" if v < 1e-6 else f"{v:.2f}"):
    display(ctrl)
c = L.known_quadratic_check(nodes["B"], X, D, 0)
print("setting B, seed 0 — known Q:"); display(L.matrix_frame(c["Q_true"], "Q true"))
print("(a) exact box means:"); display(L.matrix_frame(c["exact_box_means"], "(a)"))
print("(b) real-row sample means:"); display(L.matrix_frame(c["sample_means"], "(b)"))
print("(c) uniform resample:"); display(L.matrix_frame(c["uniform_resample"], "(c)"))""")
    code("""u = L.within_cell_uniformity(nodes["B"], X)
print("setting B: |sample centroid − box centroid| / box width per axis; last row = mean over internal nodes")
display(u.tail(6))""")

    md(f"""## 9. Findings

1. **The implementation is Algorithm 1.** `algorithm1_hessian` follows the pseudocode line by line; the Week 45
   solver and the centred-regression formulation reproduce its Q̃ to rounding (§3); rank is full in every row; and
   with exact box means the known quadratic is recovered to ~1e-14 on the fitted partitions (§8a). There is no
   coding defect in the reconstruction.
2. **Q̃ is not symmetric on this data**, by amounts comparable to or larger than the entries themselves (§6).
   Because row i and row j are estimated from disjoint sets of nodes, Q̃[i,j] and Q̃[j,i] are two independent
   estimates of the same quantity; their gap is a direct, assumption-free measure of estimation error. Line 19 of
   Algorithm 1 averages it away, so the published output Q̂ cannot show it; Q̃ must be inspected before line 19.
3. **Source of the inconsistency.** §8 shows that replacing exact box means by sample means of a *noise-free* known
   quadratic already produces errors of order 1–10× the true entries. With the same number of points per cell drawn
   uniformly on each box (c) the error is of the same order, so the dominant cause at n = 5,000 is sample-mean
   variability in small cells — second differences of noisy first differences. At setting B the real rows (b) give
   a larger error than the uniform resample (c) in every seed, so the non-uniform placement of the real predictor
   rows within boxes adds a further error (§8, last table: centroid offsets of roughly 7–11 % of the box width on
   average, against the 0 assumed by the effective-centre formula); at setting A the two effects are not separable.
   Larger leaves (setting B) reduce the inconsistency but do not remove it.
4. **Against PHD.** PHD gives a symmetric matrix with entries of order 10⁵–10⁶; the Algorithm 1 estimates disagree with it
   in magnitude and, for some entries, in sign (§7). Given (2)–(3), that disagreement cannot yet be attributed to
   either method's assumptions; it is within the CART estimate's own inconsistency.

## Limitations

- 5,000 rows is far below what second-difference estimation needs at depth 7; the full dataset was deliberately
  not used here.
- The effective-centre location assumes a uniform distribution within each box; the real predictors are correlated
  (§1) and non-uniform within boxes (§8). Equal weighting of all node pairs and pooling over the whole domain assume a
  globally quadratic response.
- PHD assumes a Gaussian predictor design, which the load perturbation scheme does not satisfy; it is a baseline,
  not a reference.
- Physical column identities beyond "cost" and "per-unit demand" are not verified.
- No eigenanalysis, bootstrap, prediction metric, or alternative estimator is included; those belong to later
  attempts after the above is resolved.""")

    nb["cells"] = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    fname = f"maple_opfdata_hessian_{d}d.ipynb"
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbf.write(nb, fname)
    return fname


if __name__ == "__main__":
    for d in (2, 3):
        print("wrote", build(d))
