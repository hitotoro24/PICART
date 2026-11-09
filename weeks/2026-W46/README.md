# Week 46 | OPFData Hessian experiments

## Studies and provenance

- **Cedar baseline (`cedar_baseline/`)**: 2D and 3D comparison of classical residual-based PHD against the existing CART effective-center reconstruction. The current concise notebooks import the shared `algorithm1_tree_hessian.py`, implement the exact all-pairs/rank/symmetrization steps, and retain the raw rows as diagnostics. The older `hessian_comparison.py` has been removed from the current directory; it remains available in Git history.
- **Maple diagnostics (`maple_diagnostics/`)**: Completed independent replication with a 50-sample minimum leaf setting and known-quadratic diagnostics. Its files and results remain separate from the Cedar baseline.
- **Future studies**: Name by actual research question and method, with a separate README documenting hypotheses, changes, reproducibility, and results. Do not add numbered `attempt_N` directories.

## Choose by study purpose, not chronological attempt number

| Attempt | 2D notebook | 3D notebook | Method guide |
| --- | --- | --- | --- |
| **Cedar baseline** | [2D](cedar_baseline/cedar_opfdata_hessian_2d.ipynb) | [3D](cedar_baseline/cedar_opfdata_hessian_3d.ipynb) | [Research guide](cedar_baseline/README.md) |
| **Maple diagnostics** | [2D](maple_diagnostics/maple_opfdata_hessian_2d.ipynb) | [3D](maple_diagnostics/maple_opfdata_hessian_3d.ipynb) | [Research guide](maple_diagnostics/README.md) |

The four notebooks retain their original saved numerical outputs and already have clear creator prefixes. The folder names indicate research purpose, **not a ranking of numerical accuracy**. An initial Cedar comparison and Maple's independent validation/diagnostics are different studies, not competing numbered attempts.

The November 9 merge reconciled concurrent GitHub changes. The Notion push workflow skips logging if required secrets are missing; a green status does not prove delivery.

## Minimal Colab experiment for Nathan

- [Known 2D quadratic: Algorithm 1](cedar_baseline/cedar_colab_quadratic_2d.ipynb), with an analytically known Hessian; no data upload needed.
- [Mounted Drive: 2D OPFData Algorithm 1 and residual PHD](cedar_baseline/cedar_colab_opfdata_2d.ipynb).

The current Cedar code imports the **same shared Algorithm 1 function** in both the quadratic and OPFData runs. It forms all same-axis effective-location pairs, checks rank, solves rows by least squares, and reports their symmetrized matrix. Historical raw asymmetry measurements in this index describe the unsymmetrized intermediate, not the final Algorithm 1 matrix.

## Current study

Use `/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv` (5,000 rows). Only the small subset is used; the full 2 GB CSV is not required. The current CSV convention is target column 0, selected demand columns 1–2 in 2D or 1–3 in 3D. Physical column meanings and units require verification against the original data-extraction procedure.

Open either notebook directly on GitHub to view the saved 2D/3D matrices, full Algorithm 1 equations, and three embedded diagnostic figures without running it. The newer version also includes a formatted comparison table and a concluding Markdown interpretation after the plots. To reproduce or change results, download or clone the repo and execute the notebook with Python/Jupyter, keeping the small CSV at the configured local path. Alternatively, run the shared implementation:

```python
from hessian_comparison import run
run(2)
run(3)
```

Install `numpy pandas matplotlib scikit-learn ipython` as needed.

## Scientific scope

The reported PHD and CART matrices are estimates, not verified true Hessians. In particular, PHD is **not** an error-free benchmark. The raw CART matrix is intentionally **not symmetrized**. Each distinct off-diagonal pair is displayed numerically as `H[i,j]`, `H[j,i]`, signed difference and absolute difference; mean, median and maximum absolute differences are summarized over unique pairs.

No eigendecomposition, physics-informed split penalty, quadratic OLS reference, bootstrap comparison, or manuscript modification is included in cedar_baseline. Matrix recovery is preliminary because the effective-center derivation has stronger assumptions than the observed OPFData design.

## Existing numerical results on 5,000-row subset (cedar_baseline)

- 2D: rank 2/2 for both tree rows; raw `H[1,2] = -733998.62` and `H[2,1] = -3154084.83`; absolute symmetry gap `2420086.20`. Maximum absolute PHD-vs-raw-CART disagreement `4735806.06`.
- 3D: rank 3/3 for all three rows. Unique off-diagonal absolute symmetry gaps: `2175983.07`, `381057.08`, `2242116.58`. Mean `1599718.91`, median `2175983.07`, maximum `2242116.58`. Maximum absolute PHD-vs-raw-CART disagreement `2557403.51`.

**These numbers are not errors against a true Hessian.** They demonstrate that the run succeeds while the raw tree reconstruction has severe symmetry inconsistency.

## Review checklist

Before any claimed improvement, verify data provenance, test the estimator on OPFData predictor locations with an analytically known quadratic response, and identify whether the resulting failure comes from geometry, nonuniform or correlated predictors, global quadratic assumptions, or code.

## Notebook rendering repair — Nov 9

Historical note: the earliest GitHub notebook files were abbreviated six-cell wrappers with **zero saved outputs**. Therefore GitHub previews showed very little of the analysis. An intermediate version subsequently saved extensive executed outputs and plots. The current version restores saved plots and formatted execution output blocks while keeping four concise code cells, including a purpose/limitations narrative, documented function definitions, actual PHD and CART matrices, row-rank diagnostics, opposing off-diagonal entry tables, mean/median/maximum symmetry gaps, with three saved heatmaps (PHD, raw CART, and asymmetry) plus a final interpretation. No estimator formulas, parameters, CSV values, or `.tex` files were modified as part of this rendering repair.

**Still outstanding:** Figures illustrating the actual left/right CART child regions for representative split nodes, and a justified empirical range of estimates from each method. These are separate visualization enhancements and are **not** claimed as part of the current rendering repair.

## Scientific interpretation of child-node means

The quadratic effective location is derived for **uniform regional means** over split boxes. CART provides **empirical child-node mean responses** from the observations assigned to each node, which need not equal the uniform regional averages. Even the first-derivative-like contrast can therefore be biased relative to the derivative at the geometric effective location when observations are not uniformly distributed. The controlled quadratic experiment in Maple diagnostics probes this gap; it does not establish an OPF ground-truth Hessian or a generally valid estimator correction.
