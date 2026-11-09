# Week 46 reader's guide — what the OPFData Hessian notebooks do and how to read them

This guide compares `chatgpt_baseline/` (initial ChatGPT comparison) and `claude_diagnostics/` (independent Claude diagnostic study), with filenames that distinguish the two executed notebook sets. It is for readers who do not want to read the code first. It explains
the question, the data, the two estimators, the checks, and how to interpret every table. Code appears only where
a few lines make the idea concrete.

---

## 1. The question in one paragraph

A power grid has demand at many buses. Solving the grid's optimal power flow (OPF) gives the cheapest way to meet
that demand, and therefore a **cost** for each demand pattern. The *gradient* of cost with respect to demand says
how much cost rises when one bus asks for a little more power. The **Hessian** goes one step further: it says
whether that answer changes when *another* bus also asks for more. Its off-diagonal entries are interactions between
buses; its diagonal entries say whether a bus's own cost is accelerating. We want to estimate this Hessian from
data alone — many (demand, cost) records — without re-solving the OPF.

Two estimators are compared:

- **CART-derived Hessian** (the method developed in Weeks 43–45): fit an ordinary regression tree, read slopes off
  its splits, and turn differences of slopes into curvature.
- **PHD** (principal Hessian directions, a classical statistical estimator from 1992): a one-formula method that
  works under an assumption about the shape of the demand distribution.

Neither is the truth. The data file contains no true Hessian. The notebooks report what each estimator gives, how
internally consistent the tree method is, and where they disagree.

---

## 2. The data

`118_ieee_smol.csv` — 5,000 rows, 100 columns, no header row.

| column | meaning | how we know |
|---|---|---|
| 0 | cost of the solved OPF (objective value) | ~7×10⁴–1.2×10⁵; rises with every demand column |
| 1 – 99 | active demand at each of the 99 load buses of the IEEE-118 test grid, in per-unit (1 p.u. = 100 MW) | the IEEE-118 case has exactly 99 loads; the column means add to 42.4 p.u. = 4,240 MW, matching the case's nominal 4,242 MW |

The notebooks use column 0 as the response and columns 1–2 (2D) or 1–3 (3D) as predictors, called *Load 1, Load 2,
Load 3*. Which physical bus each column is has not been verified; everything below is stated in terms of
"Load k", never a bus number.

The demands are **correlated** with each other (about 0.47–0.49 between any two). This matters for both estimators.

---

## 3. The CART-derived Hessian, step by step

**Step 1 — Fit an ordinary regression tree.** CART repeatedly picks one variable and one cut point that best
separates high-cost rows from low-cost rows, producing a set of boxes in demand space. We use the standard
`scikit-learn` implementation with no modifications:

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=7, min_samples_leaf=50).fit(X, y)
```

We never use the tree's *predictions*. We use its *geometry*: for each split, which variable was cut, where, and the
average cost on each side.

**Step 2 — A slope from each split.** A split along Load *s* divides a box into a left half and a right half. The
difference of the average costs, divided by half the box width, is a slope estimate:

```
gamma = 2 * (mean_right - mean_left) / (hi_s - lo_s)
```

This is the first-derivative estimate from the Week 44–45 work.

**Step 3 — Where does that slope "live"?** For a function that is quadratic, the slope estimate above equals the true
derivative at a specific point in the box, called the *effective centre*: along the split variable it is the average of
the three breakpoints `(lo_s + tau + hi_s) / 3`; along every other variable it is the box midpoint. (The 1/3 comes from
assuming the data points fill the box evenly.)

**Step 4 — Curvature from pairs of slopes.** Take any two splits on the same variable *s*. Their slopes differ, and
their effective centres differ. For a quadratic, the slope difference equals the centre difference dotted with row *s*
of the Hessian. Every such pair is one linear equation; a depth-7 tree gives tens to thousands of them; row *s* of the
Hessian is the least-squares solution. Row 1 is built only from Load-1 splits, row 2 only from Load-2 splits, and so on.

**A key consequence.** Because row *i* and row *j* are solved from *different* splits, the entry H[i,j] (from row *i*)
and the entry H[j,i] (from row *j*) are two independent estimates of the same number. A true Hessian is symmetric, so
**the gap between them is an error measurement that needs no ground truth.** Both notebooks report this gap before
anything else is concluded.

---

## 4. PHD, the baseline

Residual-based PHD does the following:

1. Centre the predictors (subtract each column's mean) and compute their covariance matrix Σ.
2. Fit an ordinary straight-line regression of cost on the predictors and keep the residuals *r* (what the line
   could not explain — for a quadratic surface, that is the curvature).
3. Average residual × predictor × predictorᵀ over the rows, call it M.
4. The estimate is Σ⁻¹ M Σ⁻¹.

```python
Xc = X - X.mean(0)
Sigma = Xc.T @ Xc / n
r = y - fitted_straight_line(X, y)
M = (Xc * r[:, None]).T @ Xc / n
H_phd = inv(Sigma) @ M @ inv(Sigma)
```

A theorem (Stein's lemma) says this equals the average Hessian **if the predictors are Gaussian**. The OPFData
demands are not Gaussian (each is a nominal value scaled by random factors), so PHD is a *baseline with an
assumption*, not a reference answer. It is always symmetric by construction, so it cannot be checked the way the
tree estimate can.

---

## 5. What the ChatGPT baseline establishes

- Re-implemented the Step 1–4 pipeline inside one script (`hessian_comparison.py`) and implemented PHD.
- Fitted the tree at depth 7 with the default minimum leaf size of 1.
- Reported the raw tree Hessian, the symmetry gap for each off-diagonal pair, and the largest difference to PHD.
- Reported in the original units (cost per p.u.²), where numbers are in the hundreds of thousands to millions.

Headline numbers (2D): H[1,2] ≈ −0.73 million, H[2,1] ≈ −3.15 million — a gap of 2.4 million, i.e. the two
estimates of the same entry disagree by more than three times the smaller one. In 3D the three pair gaps were 2.18,
0.38 and 2.24 million. chatgpt_baseline correctly concluded the raw reconstruction is severely inconsistent, and listed
possible causes (geometry, non-uniform predictors, the global-quadratic assumption, or code) without deciding.

---

## 6. What the Claude diagnostic study adds, and why

**Same method, called directly.** Instead of re-implementing Steps 1–4, claude_diagnostics imports the Week 45 package and
calls it. It checks numerically that the two implementations agree (they do, to 10⁻⁹), and reproduces chatgpt_baseline's
numbers exactly at the same tree settings. So any disagreement between attempts is about interpretation, not code.

**Two tree settings.** Setting A = chatgpt_baseline's (leaf size 1); setting B = leaf size 50. Bigger leaves mean each
average is computed from more rows and is therefore steadier.

**PHD stated precisely.** The covariance uses 1/n, the residual comes from a fit with intercept and linear terms,
and the matrix is reported as computed (its asymmetry is ~10⁻¹⁶, i.e. zero).

**Symmetry first, comparison second.** Each pair is shown as H[i,j], H[j,i], their signed and absolute gap, and the
gap relative to the larger of the two. Only after that is the tree estimate compared to PHD, entry by entry — once
raw, and once after averaging H and Hᵀ, labelled as a separate step so nothing is hidden.

**A controlled experiment to find the cause.** This is the main addition. Keep the fitted tree exactly as it is, but
replace the real cost with a *made-up quadratic whose Hessian we know*. Then run the reconstruction three ways:

| version | what each box's "average" is | what it tests |
|---|---|---|
| (a) exact box means | the exact average of the quadratic over the box, assuming points fill it evenly | code and geometry only |
| (b) real-row sample means | the quadratic evaluated at the actual data rows in the box, then averaged | adds the effect of how many rows are in each box *and* where they sit |
| (c) uniform resample | the same number of points as (b), but placed evenly at random inside the box | adds only the "how many rows" effect |

Result: (a) recovers the known Hessian to 14 decimal places in both 2D and 3D → **the code and the geometry are
correct**. (b) is wrong by 1–10× the true entries even with *no noise at all*. (c) is wrong by a similar amount. So
at 5,000 rows the dominant problem is simply that averages over small boxes are noisy, and the method takes
differences of differences of those averages. At leaf size 50, (b) is worse than (c) in every trial, so the fact
that real rows do *not* fill the boxes evenly (their centroids sit 7–11 % of the box width off centre) adds a second,
smaller error on top.

---

## 7. How to read the tables

- **Rank diagnostics.** Each Hessian row needs at least *d* independent equations; "rank = d" means it has them.
  Every row in every notebook has full rank, so the estimates exist — but existing is not the same as being
  accurate.
- **Symmetry table.** Look at the last column, the gap relative to the larger entry. Near 0 would mean the two
  independent estimates agree; values of 0.3–1.9 (what we see) mean they disagree by a third to nearly twice the
  entry size. Treat the entry as undetermined.
- **Comparison with PHD.** Differences are between two estimates. Because the tree estimate's own inconsistency is
  as large as its entries, a disagreement with PHD here says nothing yet about which method is closer to reality.
- **Controlled check.** Column (a) ≈ 10⁻¹⁴ is the proof the pipeline works. Columns (b) and (c) are the size of the
  problem on this amount of data.
- **Heatmaps.** Same numbers, in millions, coloured: blue negative, red positive, yellow–red for absolute gaps.
  They are for a quick visual read, not for extra information.

---

## 8. What the two attempts together establish

1. The reconstruction is implemented correctly and recovers a quadratic exactly when box averages are exact.
2. On the real data, at 5,000 rows, the method's estimates are inconsistent with themselves, so no Hessian entry
   can be quoted yet.
3. The reason is identified: noisy averages in small boxes (main), plus unevenly placed rows inside boxes
   (secondary).
4. PHD gives a stable, symmetric matrix, but rests on a Gaussian assumption the data violate; it is a comparison
   point, not an answer.

## 9. What would change the picture

- **More data.** The full dataset has ~800,000 rows. If the averaging noise falls as expected, the symmetry gaps
  should shrink by roughly √(800,000/5,000) ≈ 13×. This is the single most informative next run.
- **Weighting.** Currently a pair of splits backed by 25 rows counts the same as one backed by 2,500. Weighting
  by sample size is the natural fix for the main error term.
- **Accounting for uneven filling.** Using each box's actual sample centroid and spread instead of the
  even-filling assumption addresses the secondary term. This changes the derivation and belongs to a documented
  later attempt.

---

## Appendix — files

```
weeks/2026-W45/tree_hessian/        the estimator (unchanged since Week 45)
weeks/2026-W46/chatgpt_baseline/           first comparison: hessian_comparison.py + two notebooks
weeks/2026-W46/claude_diagnostics/           second comparison: hessian_tools.py, make_notebooks.py,
                                     claude_opfdata_hessian_2d.ipynb, claude_opfdata_hessian_3d.ipynb, README.md
```

To re-run claude_diagnostics from scratch: open a terminal in `weeks/2026-W46/claude_diagnostics` and run `python make_notebooks.py`
(needs `numpy pandas scikit-learn matplotlib nbformat nbclient`). To just read, open the `.ipynb` files; outputs
are saved in them.

## Naming and methodological clarification

The earlier directories `attempt_01` and `attempt_02` have been renamed to `chatgpt_baseline` and `claude_diagnostics`. This avoids implying that Claude's study supersedes the ChatGPT comparison or that a smaller raw symmetry gap establishes accuracy. Both remain separately documented in Git history. **Regional-mean caveat:** the effective location for a general quadratic follows from exact uniform averages over rectangular children. An empirical CART child-node mean response is different when observation locations are uneven within those regions, potentially changing the derivative-like contrast itself. This issue arises before the row-wise Hessian fit.
