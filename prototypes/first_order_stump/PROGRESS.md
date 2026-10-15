# First-order PICART stump — progress

A side study, not the main project. It tests the proposal's first-order
algorithm (§5.2, Algorithm 2, Eqs. 10–11) in the simplest setting: **one input
variable and a single split** (a "stump"), on noisy data from `f(x) = exp(3x)`,
which obeys the first-order equation `f'(x) = 3 f(x)`. A standard
scikit-learn CART stump is the baseline.

**Scope:** only depth 1 and one input dimension. This is not the full recursive
algorithm, and not a replication of the proposal's Figure 2.

## How to run

From the repository root:

```bash
python -m unittest discover -s prototypes/first_order_stump/tests -v
python prototypes/first_order_stump/picart_experiment.py --seed 17 --n 120 --sigma 1.5 --lambdas 0,0.01,0.1,1,10,100
```

Outputs go to `experiments/` in this folder: `fit.png`, `lambda_sweep.png`, and
`latest.json` (seed, parameters, metrics). Before committing a run, fill in
`interpretation` and `next_step` in `latest.json`, and add an entry to
`experiments/EXPERIMENT_LOG.md`.

## Files

- `picart_experiment.py`: the stump: candidate-split search, closed-form child
  values, plots, metrics.
- `tests/test_picart_stump.py`: checks that λ=0 gives the ordinary CART result,
  that the size-weighted average of the child values is preserved, and that the
  physics residual shrinks as the penalty grows.
- `experiments/EXPERIMENT_LOG.md`: experiment journal (hypothesis →
  observation → interpretation → next step).

## Issues found in the proposal (checked against the PDF)

1. **Eq. (11) swaps N_l and N_r.** Setting the derivative of Eq. (10) to zero
   gives `(N_l + 4λ/w²)·μ_l − (4λ/w²)·μ_r = N_l·ȳ_l − 2λφ/w` (and symmetrically
   for μ_r). The printed matrix pairs `N_r` with `μ_l`. The code uses the
   derived form.
2. **The sign of φ in Algorithm 2 looks reversed.** For Eq. (10)'s residual
   `2(μ_r−μ_l)/w − φ` to equal `Df − h`, the target must be
   `φ = (h − Σ_{p'≠p} δ_{p'} g_{p'}) / δ_p`. The printed line is
   `Σ_{p'≠p} δ_{p'} g_{p'} − h`, without the division by `δ_p`. The code uses
   `φ = +3·(parent prediction)`.
3. **Reversed assignment in Algorithm 2:** it prints `state cost ← best cost`,
   but it should be `best cost ← state cost`. The code does this correctly.

Issue 2 bears directly on the general first-order question: if `δ_p = 0`,
splitting on variable `p` cannot change the physics residual.

## Modelling assumptions to keep in mind

- φ uses the **parent's mean response** as a stand-in for `f` at the node
  midpoint. For a curved `f` such as `exp(3x)`, `3·mean(y)` differs from
  `3·f(midpoint)`, so this is an approximation that can be tested.
- φ is held **fixed** during the split search, not solved jointly with the
  child values.
- Node bounds are the known domain `[0, 1]`, not the sample's minimum and maximum.

## Log

| Date | Change / finding |
|---|---|
| 2026-10-15 | Imported into the repo. All tests pass; script runs. Default run (seed 17): as λ grows, the left child rises (3.59 → 3.74) and the right child falls (12.51 → 11.95), the **opposite** of the initial hypothesis. The CART slope (≈17.8) is above the physics target, so the penalty flattens the step. Interpretation pending. |
