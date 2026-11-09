# Week 46 | OPFData Hessian experiments

## Sequence

- **attempt_01**: 2D and 3D comparison of classical residual-based PHD against the existing CART effective-center reconstruction. Implemented in `hessian_comparison.py`, with separate runnable notebook entry points.
- **attempt_02**: Reserved for an independently executed Claude Code review and implementation. Do not overwrite attempt_01 or fabricate results attributed to Claude.
- **attempt_03** and later: Subsequent experiments only after documenting the change, hypothesis, and results of previous attempts.

## Current study

Use `/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv` (5,000 rows). Only the small subset is used; the full 2 GB CSV is not required. The current CSV convention is target column 0, selected demand columns 1–2 in 2D or 1–3 in 3D. Physical column meanings and units require verification against the original data-extraction procedure.

Run a notebook from this directory in Jupyter or VS Code, or run the following from a Python session:

```python
from hessian_comparison import run
run(2)
run(3)
```

Install `numpy pandas matplotlib scikit-learn ipython` as needed.

## Scientific scope

The reported PHD and CART matrices are estimates, not verified true Hessians. In particular, PHD is **not** an error-free benchmark. The raw CART matrix is intentionally **not symmetrized**. Each distinct off-diagonal pair is displayed numerically as `H[i,j]`, `H[j,i]`, signed difference and absolute difference; mean, median and maximum absolute differences are summarized over unique pairs.

No eigendecomposition, physics-informed split penalty, quadratic OLS reference, bootstrap comparison, or manuscript modification is included in attempt_01. Matrix recovery is preliminary because the effective-center derivation has stronger assumptions than the observed OPFData design.

## Existing numerical results on 5,000-row subset (attempt_01)

- 2D: rank 2/2 for both tree rows; raw `H[1,2] = -733998.62` and `H[2,1] = -3154084.83`; absolute symmetry gap `2420086.20`. Maximum absolute PHD-vs-raw-CART disagreement `4735806.06`.
- 3D: rank 3/3 for all three rows. Unique off-diagonal absolute symmetry gaps: `2175983.07`, `381057.08`, `2242116.58`. Mean `1599718.91`, median `2175983.07`, maximum `2242116.58`. Maximum absolute PHD-vs-raw-CART disagreement `2557403.51`.

**These numbers are not errors against a true Hessian.** They demonstrate that the run succeeds while the raw tree reconstruction has severe symmetry inconsistency.

## Review checklist

Before any claimed improvement, verify data provenance, test the estimator on OPFData predictor locations with an analytically known quadratic response, and identify whether the resulting failure comes from geometry, nonuniform or correlated predictors, global quadratic assumptions, or code.
