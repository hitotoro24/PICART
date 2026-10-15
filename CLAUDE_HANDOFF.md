# Claude handoff — PICART RA experiment, not the F1 project

You are working on the author's RA project on first-order Physics-Informed CART
(PICART), based on Nathan Wycoff's project proposal, Section 5, Algorithm 2,
Equations (9)–(11), and the toy exponential experiment (§5.2.1).

**Author ownership:** The user defines the hypothesis, pseudocode changes,
research interpretation, and conclusions. Do not quietly rewrite their notebook,
solve every exercise, or alter the project's scientific claims. Use the provided
`picart_experiment.py` as a *separate scratch experiment*, never overwrite the
user's `research.ipynb`.

## Research hypothesis

The user initially predicts that increasing the physics penalty decreases the
left child prediction and increases the right child prediction. Treat this as
an empirical/conditional hypothesis, NOT a theorem. For a fixed split,
movement direction is determined by whether the CART derivative estimate is
above or below the target derivative. A larger penalty reduces the absolute
physics residual for a fixed split, but may or may not improve test MSE.

## Source-aligned algorithm

For each current tree node i:
  For each feature p:
    Form `phi` using the differential equation and the *existing* tree/node
    estimates, as in Algorithm 2. Do not use ground-truth f in phi.
    For candidate threshold tau:
      Partition observations into left/right children.
      Compute N_l,N_r, empirical means and node width w.
      Solve Eq. (10) in mu_l,mu_r with the correctly-derived FOCs.
      Calculate the **full** split score: both child RSS + physics penalty.
      Save the best variable, threshold, child predictions.
  Update/split node and repeat recursively when moving beyond depth 1.

**Current scope:** Only the depth-1, one-dimensional special case is implemented.
Do not describe it as full Algorithm 2. scikit-learn `DecisionTreeRegressor`
is a data-only CART baseline; it cannot be passed a custom PICART split cost
through the normal public API. Any deeper PICART implementation will require
custom tree growth or an explicitly limited alternative.

## Mathematical checks

- The printed Algorithm 2 target `phi = sum(other derivative terms) - h(...)`
  appears to have the wrong sign relative to the target-derivative definition
  in Eq. (10). In 1D `f'=rate*f`, use the physically required `phi=+rate*f`
  based on the *parent's fitted estimate*, not the negative; document this
  discrepancy and ask the advisor if it matters for the multidimensional code.
- The printed proposal Eq. (11) appears inconsistent with Eq. (10). Our correct
  system for fixed phi is `[[N_l+a, -a],[-a,N_r+a]]` where `a=4*lambda_tilde/w**2`,
  with RHS `[N_l*ybar_l - 2*lambda_tilde*phi/w,
            N_r*ybar_r + 2*lambda_tilde*phi/w]`.
- Verify lambda=0 returns the CART child sample means.
- For fixed split, `N_l*mu_l + N_r*mu_r = sum(y)`.
- For fixed split and fixed phi, as lambda rises, absolute residual should
  not increase (within numeric precision).
- Reoptimized *split locations* can jump as lambda changes. Separate those
  from fixed-split effects.
- The exponential test `f(x)=exp(rate*x)` obeys `f'(x)=rate*f(x)`, making
  first-order physics available while having varying slopes and noisy data.
  The initial code uses `phi=rate*mean(y_in_parent)` at the root as the existing
  parent prediction, so it is not an oracle derivative and may be biased.

## Requested collaboration

1. Preserve the existing file structure and tests unless asked otherwise.
2. Inspect one issue/change at a time, propose a small patch, and explain why.
3. Make `picart_experiment.py` runnable, preserving assumptions and results.
4. Generate actual PNG visualizations, and keep machine-readable metrics in
   `experiments/latest.json` with the seed and parameters recorded.
5. Let the user write `interpretation` and `next_step` in the JSON/notes before
   pushing a substantive experiment.
6. Do NOT claim a GitHub push automatically updates Notion until the
   GitHub Actions workflow has run successfully with repository secrets.
7. Do NOT copy the prior F1 Notion experiment log into the PICART workspace.

## First code review question

Inspect how `phi` is evaluated in the exponential example, and explain why
`rate*mean(y)` can differ from `rate*f((l+u)/2)`. Do not silently change it.
