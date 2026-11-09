# Archived original research brief (before the independent Attempt 02)

This document preserves the planning brief prepared before the independent implementation was committed. **It is historical context, not a current status report or description of the executed Claude results.** The independent Attempt 02 has since been pushed; see [its own README](../attempt_02/README.md) and notebooks for the actual implementation, comparisons, and findings. Statements below about work being "not yet executed" or awaiting review describe the status *when this brief was drafted*. No files inside `attempt_02/` are changed by recording this brief.

---

# Independent OPFData Hessian comparison: attempt 02

**Documentation and implementation brief | Not yet executed**

This document specifies the expected structure of an independently implemented follow-up to Week 46 Attempt 01. **No implementation, new results, or commit by Claude is claimed here.** The previous attempt remains unchanged.

## Research objective

Independently reproduce a two-dimensional and three-dimensional comparison between a CART-derived Hessian estimator and residual-based Principal Hessian Directions (PHD), using the same observations and assumptions documented in Attempt 01. Test whether implementation details explain any numerical differences. Preserve numerical failures instead of hiding them through symmetrization or undocumented tuning.

## Inputs and comparison

- Dataset: `/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv`, 5,000 rows, no header.
- Provisional response: CSV column 0.
- 2D predictors: columns 1 and 2. 3D predictors: columns 1, 2 and 3.
- Start with the same untransformed data and CART settings (`max_depth=7`, `random_state=0`). Treat modifications as distinct experiments with named hypotheses and retained prior outputs.
- Existing method and test reference: `weeks/2026-W45/tree_hessian/uneven.py` and its associated tests.
- Attempt 01 reference: `weeks/2026-W46/attempt_01/`.
- Do not treat the PHD matrix as true Hessian; neither baseline is ground truth on this dataset.

## Deliverables

1. A self-contained executed 2D notebook and 3D notebook in a new `attempt_02` folder.
2. A reader-facing `README.md` that can be understood before opening either notebook. Organize it into research question, dataset, both algorithms, execution, numerical findings, interpretation, limitations, and next diagnostic.
3. For every implemented function, concise docstrings specifying **purpose**, **arguments**, and **returns**. Place a brief research-purpose Markdown introduction before substantive code sections. Avoid provenance stories or authorial claims in research narrative.
4. Direct numerical display of PHD and **raw unsymmetrized** tree matrices. For each distinct pair `i<j`, show `H[i,j]`, `H[j,i]`, signed difference, and absolute difference. Compute mean, median and maximum absolute pair gaps; specify how many pairs contribute to each summary.
5. Numerically annotated figures with readable axis labels and consistent scale where possible. Include a **representative actual CART split** showing the observations, left/right child regions, sample counts, child mean costs, split threshold, and first-order contrast. Use an appropriate 2D slice/projection for 3D. Do not depict illustrative or invented data as real observations.
6. When using *ranges of repeated estimates* for PHD and CART, first define the repetition protocol (e.g. independently repeated subsamples with fixed sizes and recorded random seeds). Explain what variation is summarized. Do not present those ranges as confidence intervals without a statistical argument.
7. Independent verification of the PHD formula and covariance scaling, plus a check that the tree-row algebra reproduces the existing reference implementation on a controlled quadratic input before attributing a discrepancy to data or theory.
8. A short comparison with Attempt 01: what was reproduced, what changed, whether the raw symmetry and PHD disagreement improved, and what remains unidentified.

## Required scientific cautions

- Do not silently standardize variables, change root-region definitions, alter tree-depth settings, or replace the effective-center formula.
- Do not add extra models or error metrics without an explicit question and interpretation.
- Do not use `max(abs(H_cart-H_phd))` as an error to ground truth; it is a measure of **between-method disagreement**.
- Do not claim that a full-rank least-squares system establishes an accurate Hessian.
- Do not invent the physical bus mapping, cost units, or an OPF ground-truth Hessian from an unlabeled CSV.
- Do not modify the conference paper, original CSV, prior attempts, or unrelated research.

## Work sequence and review gate

**Phase A: inspect only.** Read both existing methods and the source notebooks. Write down the formulas they actually implement, including how tree root bounds are set. Check internal consistency and report concerns.

**Phase B: reproduce.** Implement separately, run with the small CSV, and preserve printed matrices and numerical diagnostics. Output must be executable and should have saved figures so GitHub's notebook preview renders the entire analysis.

**Phase C: interpret.** Explain whether both estimates are numerically stable enough to interpret, whether the observed raw CART asymmetry persists, and what controlled test would isolate its cause. Avoid claiming an improvement solely because a symmetric matrix was produced.

**Phase D: review before upload.** Present the complete files, the diff against Attempt 01, and numerical evidence for inspection. **Do not commit or push anything until the researcher explicitly approves.**

## Copy-ready instructions for Claude Code

```text
Read README_Week46_Attempt01_DRAFT.md and READER_GUIDE_Attempt02_CLAUDE_DRAFT.md, along with the current files in hitotoro24/PICART at weeks/2026-W45/ and weeks/2026-W46/attempt_01/.

Make an independent attempt_02 to reproduce PHD and the tree-derived Hessian on the 5,000-row OPFData subset for 2D and 3D. Explain the methods in plain professional research prose. The notebooks must be self-contained and executed with saved outputs, include docstrings for every function, show actual left/right CART child regions and numerical symmetry discrepancies, and clearly distinguish estimator disagreement from error to a true Hessian.

Do not modify previous attempts, the data, or the manuscript. Do not push or commit anything. Present code, notebooks, a reader-facing README, an explicit change log, and verified results for review first.
```

**Status:** Awaiting independent execution and approval. There are no Claude-generated findings to report in this document.