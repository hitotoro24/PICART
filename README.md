# PICART | Physics-Informed Regression Trees

**Research status:** exploratory PhD research; **not** a validated physics-informed tree library or a production-ready Hessian estimator. Developed with Dr. Nathan Wycoff at UMass Amherst.

PICART investigates whether derivatives inferred from regression-tree partitions can be used to enforce differential-equation constraints during tree construction. The current repository concentrates on the prerequisite problem: **estimating first- and second-order derivatives from tree geometry and regional response averages**. The physics-penalized training algorithm is not implemented as a general system.

## Research question

A CART prediction is piecewise constant, so direct differentiation of its prediction function is uninformative. Following *Regression Trees Know Calculus*, we investigate derivative-like contrasts formed from child-node response means. For a quadratic target, contrasts from multiple splits can be assembled into linear systems for Hessian rows.

**Evidence to date:** on known quadratic functions, controlled experiments recover Hessian entries with exact regional means; finite-sample node means introduce error. On the OPFData pilot, empirical CART-derived Hessians exhibit substantial inconsistency between independently estimated off-diagonal entries. These findings do **not** validate an OPF Hessian estimator or a general smooth-function theory.

## Start here

| Goal | Entry point |
| --- | --- |
| Understand the motivation and research history | This README and the [weekly study index](#research-log) |
| Reproduce controlled 3D quadratic checks without external data | [Week 45 progress and commands](weeks/2026-W45/PROGRESS.md) |
| Inspect the current OPFData study | [Week 46 overview](weeks/2026-W46/README.md) and [reader's guide](weeks/2026-W46/GUIDE.md) |
| Inspect the shared Week 46 Algorithm 1 implementation | [Cedar implementation](weeks/2026-W46/cedar_baseline/algorithm1_tree_hessian.py) |
| Review the first-order proof-of-concept | [One-split experiment](prototypes/first_order_stump/PROGRESS.md) |

## Installation and a reproducible starting experiment

Use Python 3 with an isolated environment. Dependencies are declared in [requirements.txt](requirements.txt); Python/package versions have not been fully locked.

```bash
git clone https://github.com/hitotoro24/PICART.git
cd PICART
python3 -m venv .venv
source .venv/bin/activate             # macOS/Linux
python -m pip install -r requirements.txt

# Core and first-order prototype tests
python -m unittest discover -s tests -v
python -m unittest discover -s prototypes/first_order_stump/tests -v

# Controlled Hessian experiment: no OPFData CSV required
cd weeks/2026-W45
python -m pytest -q tree_hessian/tests
python experiments/run_controlled.py
```

The controlled script writes output in `weeks/2026-W45/experiments/results/`; inspect [the Week 45 report](weeks/2026-W45/report_3d_hessian.md) for test design, the specified quadratic, reported error, sampling regimes, and unresolved assumptions. The reported exact-mean error is near floating-point precision for that quadratic and geometry. **These are documented historical results, not test results independently rerun for this README update.**

**Week 46:** Start with [the data-free synthetic 2D notebook](weeks/2026-W46/cedar_baseline/cedar_colab_quadratic_2d.ipynb). The real-data notebooks require the external `118_ieee_smol.csv` file and a local-path adjustment described in the [Cedar guide](weeks/2026-W46/cedar_baseline/README.md). The external dataset is not bundled; bus-to-column mapping and some physical units remain unverified. Do not treat the saved OPFData output as a ground-truth sensitivity result.

Windows users can activate the environment with `.venv\\Scripts\\activate` or use an equivalent environment manager.

## Methods and interpretation

- **Controlled 2D/3D quadratics:** compare known Hessians with estimators based on exact box averages and sampled node means.
- **Actual CART geometry:** extract split axes, bounds, thresholds, and node values; check whether the equations identify each row.
- **Uneven splits:** use quadratic effective locations and same-axis split pairs, solve row-wise least squares, inspect rank/conditioning and asymmetry, then symmetrize the reported matrix.
- **OPFData pilot:** compare CART reconstruction against residual-based principal Hessian directions (PHD) as a *baseline*, not as truth.

The effective-location identity depends on a quadratic target and **uniform regional averages**. A fitted tree's empirical child means need not estimate those averages accurately when covariates are correlated, nonuniform, or sparsely represented in small nodes. Pair weighting, selection of local versus global contrasts, statistical uncertainty, and extensions beyond quadratics are open research questions.

## Repository map

| Location | Contents |
| --- | --- |
| `weeks/2026-W42` through `weeks/2026-W46` | Chronological research notes, notebooks, experiments and outputs |
| `weeks/2026-W45/tree_hessian/` | Controlled estimators, tree extraction, uneven-split reconstruction and tests |
| `weeks/2026-W46/cedar_baseline/` | Shared Algorithm 1 implementation and 2D/3D OPFData comparison |
| `weeks/2026-W46/maple_diagnostics/` | Independent diagnostic implementation, alternative leaf size and controlled checks |
| `prototypes/first_order_stump/` | Limited first-order PICART experiment |
| `tests/` | Root utility checks; these are not a complete research regression suite |
| `.github/workflows/` | Notion push logging; **not** a test CI workflow |

## Research log

| Period | Primary result or question | Evidence |
| --- | --- | --- |
| [Week 46, Nov 9–15](weeks/2026-W46/README.md) | OPFData/PHD comparison and tests of estimator disagreement, assumptions and numerical consistency | [Reader's guide](weeks/2026-W46/GUIDE.md) |
| [Week 45, Nov 2–8](weeks/2026-W45/PROGRESS.md) | 3D quadratic, controlled stencils, actual CART geometry and effective-centre solver | [Detailed report](weeks/2026-W45/report_3d_hessian.md) |
| [Week 44](weeks/2026-W44/PROGRESS.md) | Effective locations and 2D-to-D Hessian construction | Week 44 notebook |
| [Week 43](weeks/2026-W43/) | Stencils and unequal-split investigations | Weekly notes |
| [Week 42](weeks/2026-W42/) | First-order proposal review and initial one-split prototype | [Prototype progress](prototypes/first_order_stump/PROGRESS.md) |

## Current limitations

1. A quadratic exact-average identity is **not** a consistency theorem for a fitted CART estimator on general functions.
2. OPFData results are estimates without a verified true Hessian; raw off-diagonal asymmetry is a warning about internal inconsistency.
3. The full empirical data provenance and units, cross-machine setup and statistical uncertainty require additional work.
4. Passing local tests does not imply validated physics-informed training. The repository does not yet provide a stable API, benchmark suite or continuous test pipeline.

## References

- Nathan Wycoff, *Regression Trees Know Calculus*, [arXiv:2405.13846](https://arxiv.org/abs/2405.13846).
- Nathan Wycoff, *Physics-Informed Regression Trees*, research proposal.

This is an active research repository. Reuse experimental code only after reviewing the assumptions and reproducing the relevant diagnostic.
