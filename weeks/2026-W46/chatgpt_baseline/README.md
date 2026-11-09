# Estimating cost curvature from regression-tree splits

**OPFData IEEE-118 | Week 46 | ChatGPT baseline comparison | 2D and 3D pilot**  
**Status:** Preliminary analysis. This document and the associated notebooks are research artifacts, not a validated power-system sensitivity study.

## Research question

Can the mean-response differences at regression-tree splits be used to reconstruct second-order information about a regression function? On real optimal-power-flow data, how does the resulting matrix compare with a classical residual-based Principal Hessian Directions (PHD) estimate?

The initial experiments use two and three selected demand variables. Neither estimator is treated as the true Hessian. The purpose is to check that the implementations run, inspect the numerical matrices, and identify discrepancies that require further study.

## Data and scope

The input is `118_ieee_smol.csv`, a 5,000-row, headerless subset attributed to the OPFData IEEE-118 dataset. The current extraction convention treats **CSV column 0 as the optimal operating-cost response**, and **columns 1, 2, and 3 as candidate demand predictors**. The 2D experiment uses columns 1–2; the 3D experiment uses columns 1–3. These numerical columns are indexed from zero. The mapping of each selected demand column to a physical bus, the units, and the details of the CSV extraction have **not yet been independently verified**.

Each experiment uses the original CSV values without standardization. The fitted tree has `max_depth=7` and `random_state=0`. The full approximately 2 GB CSV is not used. The same observations are passed to both methods.

**What is being estimated?** Because only two or three of 99 candidate demand variables are retained, these experiments concern the observed relationship between cost and the *selected* variables, with other inputs varying in the data. This is not automatically the same as varying only those selected demands in the physical OPF model while fixing all others.

## Two approaches to curvature

### 1. Residual-based Principal Hessian Directions (PHD)

A straight-line regression describes the main trend between demand and cost but cannot represent curvature. PHD looks for structured patterns in the part of cost that remains after subtracting this linear trend.

For each observation, let $z_i=x_i-\bar x$ be its centered selected demands, and let $r_i=y_i-\widehat y_i$ be the residual from ordinary linear regression. The residual-based PHD matrix used here is

$$
\widehat H_{\mathrm{PHD}} =
\widehat\Sigma^{-1}
\left(\frac{1}{n}\sum_{i=1}^n r_i z_i z_i^\top\right)
\widehat\Sigma^{-1},
\qquad
\widehat\Sigma=\frac{1}{n}\sum_{i=1}^n z_i z_i^\top.
$$

In words: the weighted products $z_i z_i^\top$ describe which pairs of demand values are associated with positive or negative deviations from a straight-line cost model. The covariance matrices account for the predictors' scales and correlations. Under classical Gaussian-predictor and regularity assumptions, the population expression has an average-Hessian interpretation. On these data it remains an **estimated reference method**, not ground truth.

In the notebook, `estimate_phd(X, y)` returns the matrix and the predictor-covariance condition number. It does not fit a regression tree or rerun the AC-OPF solver.

### 2. Tree-derived Hessian reconstruction

An ordinary CART regression tree splits the demand space to reduce the within-region sum of squared response errors. Each internal node has a splitting variable $s$, a threshold $\tau$, rectangular bounds $[\ell,u]$, and two child-node mean costs $\bar y_L$ and $\bar y_R$.

A CART predictor is piecewise constant, so differentiating its predictions directly would not produce the desired curvature. Instead, the existing method extracts a **first-order contrast** from each split:

$$
\gamma_s = \frac{2(\bar y_R-\bar y_L)}{u_s-\ell_s}.
$$

The contrast uses only the two child-node response means and the width of their parent region along variable $s$. The method assigns that contrast an *effective location* based on the node's bounds and threshold:

$$
 c_s=\frac{\ell_s+\tau+u_s}{3},
 \qquad c_k=\frac{\ell_k+u_k}{2}\quad(k\ne s).
$$

For an appropriate quadratic function with the required regional-averaging assumptions, the contrast equals the relevant partial derivative at $c$. Because a quadratic's gradient is linear, differences between contrasts at two nodes splitting on the same variable provide linear equations for one Hessian row. The notebook solves those equations by least squares and repeats the procedure for each variable.

The functions are arranged in the notebook as follows:

| Function | Input | Output | Role |
| --- | --- | --- | --- |
| `load_opfdata` | CSV path and selected column numbers | $X$, $y$, loaded path | Read the pilot data without changing its values. |
| `estimate_phd` | $X$, $y$ | PHD matrix, covariance condition number | Calculate the comparison estimate from residuals. |
| `collect_splits` | Fitted CART, assumed outer bounds | Internal split records | Retrieve split variable, threshold, region bounds, and child mean responses. |
| `estimate_tree_hessian` | Split records, number of variables | Raw matrix, row diagnostics | Build and solve the derivative-contrast equations. |
| `off_diagonal_checks` | Raw estimated matrix | Pairwise table and summary | Measure differences between independently estimated opposing entries. |
| `plot_hessian` | Matrix and title | Annotated figure | Display matrix values for inspection. |

**Important:** The method treats empirical data bounds as provisional outer tree-region bounds and solves Hessian rows independently. Its quadratic derivation does not guarantee valid Hessian recovery for the real, possibly nonuniform and correlated demand data.

## Reproducibility

Open either of the two self-contained executed notebooks in Jupyter or VS Code:

- [`chatgpt_opfdata_hessian_2d.ipynb`](https://github.com/hitotoro24/PICART/blob/main/weeks/2026-W46/chatgpt_baseline/chatgpt_opfdata_hessian_2d.ipynb)
- [`chatgpt_opfdata_hessian_3d.ipynb`](https://github.com/hitotoro24/PICART/blob/main/weeks/2026-W46/chatgpt_baseline/chatgpt_opfdata_hessian_3d.ipynb)

Requirements: Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, Jupyter/IPython. The executed notebooks record Python 3.13.5, NumPy 2.3.5, and pandas 2.2.3; other package versions are not pinned in the notebooks.

The notebook currently expects the small CSV at:

```text
/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv
```

If working on another computer, change `CSV_PATH` in the first code cell. Run all cells from the top. Verify the printed observation count (5,000), tree depth, split count, row ranks, PHD matrix, raw CART matrix, and numerical symmetry table. Results already saved inside the notebook can be viewed on GitHub without running anything.

The notebooks use **raw, unstandardized inputs**. Any future change of units or scaling must be recorded because Hessian entries depend on the units of both the inputs and the response.

## Numerical findings

### 2D pilot

Residual-based PHD matrix, rounded to two decimal places:

$$
\widehat H_{\mathrm{PHD}} =
\begin{pmatrix}
 140086.44 & -474727.79 \\
 -474727.79 & 830031.37
\end{pmatrix}.
$$

Raw tree-derived matrix:

$$
\widehat H_{\mathrm{tree}} =
\begin{pmatrix}
 210716.98 & -733998.62 \\
 -3154084.83 & -3905774.69
\end{pmatrix}.
$$

- Tree: 126 internal splits; both Hessian rows had full rank.
- The two independently estimated off-diagonal entries are $\widehat H_{12}=-733998.62$ and $\widehat H_{21}=-3154084.83$.
- The absolute gap between them is approximately **2,420,086.20**.
- Largest absolute entrywise difference between **PHD and raw CART** is approximately **4,735,806.06**.

### 3D pilot

Residual-based PHD matrix:

$$
\widehat H_{\mathrm{PHD}} =
\begin{pmatrix}
 152100.92 & -232667.53 & -106877.38 \\
 -232667.53 & 956292.50 & -353929.23 \\
 -106877.38 & -353929.23 & 249668.28
\end{pmatrix}.
$$

Raw tree-derived matrix:

$$
\widehat H_{\mathrm{tree}} =
\begin{pmatrix}
 -399605.64 & 1153309.03 & 1158807.55 \\
 -1022674.04 & -1601111.01 & 120826.50 \\
 777750.47 & -2121290.08 & 315482.40
\end{pmatrix}.
$$

- Tree: 124 internal splits; all three Hessian rows had full rank.
- There are three distinct off-diagonal pairs. Their absolute opposing-entry gaps are approximately **2,175,983.07**, **381,057.08**, and **2,242,116.58**.
- Across those three pairs, the mean gap is **1,599,718.91**, the median gap **2,175,983.07**, and the maximum gap **2,242,116.58**.
- Largest absolute entrywise difference between **PHD and raw CART** is approximately **2,557,403.51**.

These reported differences are **not errors relative to a true Hessian**. There is no independently known Hessian in this CSV. The PHD and CART estimates may reflect different assumptions and, for a nonquadratic underlying relationship, different curvature summaries.

## How to read the results

The notebook displays the PHD and raw CART matrices, a table of opposing off-diagonal entries, and annotated matrix heatmaps. The heatmap numbers are shown in **millions of the original matrix-entry units** for legibility; the displayed matrix tables contain the original numerical values. These values should not be given physical units until the CSV extraction and cost convention are confirmed.

The symmetry diagnostic is **not** a Boolean test such as `H[i, j] == H[j, i]`. For every unique pair $(i,j)$ with $i<j$, it reports both estimates, their signed difference, and their absolute gap:

$$
 d_{ij}=\widehat H_{ij}-\widehat H_{ji},\qquad a_{ij}=|d_{ij}|.
$$

Mean, median and maximum symmetry gaps summarize the unique off-diagonal pairs. In 2D, there is one pair, so all three summaries coincide. In 3D, there are three pairs. **These summaries describe disagreement among matrix entries, not variability across repeated datasets or tree fits.**

No symmetrization is applied before calculating these gaps. Averaging $\widehat H_{ij}$ with $\widehat H_{ji}$ would force symmetry but would not explain the underlying inconsistency.

## Limitations and next experimental decisions

1. **Data provenance.** The current CSV column interpretation and physical units need confirmation against the OPFData extraction script.
2. **Tree-regional averaging.** The effective-center identity is derived under conditions that need not hold for observations with nonuniform or correlated demands.
3. **Global versus local curvature.** The tree procedure fits one Hessian matrix across all observed nodes. Real optimal operating cost can have curvature that changes with loading conditions and active constraints.
4. **PHD assumptions.** Residual-based PHD is not a known true Hessian. Classical interpretation depends on the predictor distribution and regularity assumptions.
5. **Numerical instability.** Full rank means the row-wise least-squares problems are identifiable, not necessarily well-estimated. Large off-diagonal discrepancies remain in the raw tree matrix.
6. **Presentation gaps.** The saved figures display matrix values but do **not yet** show representative left/right CART child regions or empirical ranges of repeated PHD/CART estimates. Such ranges must be computed from a specified repetition procedure before being plotted; they must not be labeled confidence intervals by default.

**Follow-up diagnostic, not part of this baseline:** Retain the OPFData predictor positions, generate responses from a known quadratic, and run the unchanged CART estimator. Since the quadratic's Hessian is then known, this would test whether the observed predictor distribution alone can cause reconstruction error. A related controlled experiment is now recorded separately in the Claude diagnostics study; it was **not performed in this baseline**.

## Naming and provenance

This folder is the **ChatGPT baseline comparison**. It is not a numbered attempt or an accuracy ranking. For a separate independent replication with additional controlled tests and a larger minimum leaf size, see [Claude diagnostics](../claude_diagnostics/README.md). The notebooks preserve their saved results. The original `attempt_01` directory name remains traceable in Git history.

## Research status

ChatGPT baseline establishes that both estimation pipelines can be executed on the small OPFData subset. It also reveals substantial disagreement between the methods and, independently, within the raw tree-derived matrix. These findings justify further validation; they do not establish that either estimator recovers true optimal-power-flow curvature. The conference manuscript has not been updated as part of this documentation.