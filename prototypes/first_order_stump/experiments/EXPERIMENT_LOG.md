# PICART experiments — research notes

Record conclusions **before** treating the experiment as evidence. Code and parameters
are versioned by Git commit; Notion automatically links that exact version after
pushes once the workflow is configured.

## Experiment 001: exponential regression stump

**Question:** Does adding a correct first-order differential penalty improve predictions?

**Pre-run hypothesis (yours):** Increasing the penalty pushes the left child down and the right child up.

**Important qualification to test:** The sign of the movement depends on the
physics derivative target relative to CART's initial estimated derivative.

**Design choices to decide:** sample count, Gaussian noise level, seed, penalty grid,
fixed split versus split optimization; then explain any deviations from the proposal.

**Observations:** (fill after your run)

**Plot interpretation:** (fill after examining `fit.png` and `lambda_sweep.png`)

**Limitations and next experiment:** (fill)
