# PICART — first-order physics-informed regression-tree prototype

**Separate research project** from the F1 pit-strategy surrogate.

## What each file does

- `picart_experiment.py`: exploratory first-order depth-one PICART model, using
  Algorithm 2's split-search pattern. `DecisionTreeRegressor` supplies a CART
  baseline. Fits noisy `exp(3x)` data, tests different penalties, saves images
  and metrics to `experiments/`.
- `CLAUDE_HANDOFF.md`: explains the mathematical assumptions and exactly how
  Claude should collaborate without taking over your notebook or reasoning.
- `experiments/EXPERIMENT_LOG.md`: your short experiment journal; add the
  hypothesis, actual interpretation, limitations, and next step each run.
- `tools/notion_push_log.py`: sends a new Notion *subpage* for each GitHub push,
  with commit URL, change summary, optionally *newly committed* experiment
  metrics, and uploaded PNG plots. It never invents missing measurements.
- `.github/workflows/notion-on-push.yml`: triggers the Notion script on push.
- `tests/test_picart_stump.py`: small mathematical correctness checks.

## Local start

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python picart_experiment.py --seed 17 --n 120 --sigma 1.5 --lambdas 0,0.01,0.1,1,10,100
```

Open `experiments/fit.png` and `experiments/lambda_sweep.png`. Read
`experiments/latest.json`, then write your own interpretation and next step.
Commit the updated JSON and PNG figures to keep results reproducible.

The script is **not** the paper's full recursive implementation or a verified
Figure-2 replication. The proposal describes a 30-replicate example without
all simulation details; our initial defaults are deliberately explicit and
illustrative. Extending beyond a stump requires further algorithm choices.

## GitHub → Notion, on every push (not on an hourly schedule)

1. Put these files in a **separate private repository** for your RA research.
   Confirm advisor/institution release policies before using any public repo.
2. In Notion, make a parent page such as `PICART — Experiment Log`, then give
   a Notion integration edit/content access to that page.
3. On GitHub go to repository **Settings → Secrets and variables → Actions**.
   Add exactly two **repository secrets**, never commit their values:
   - `NOTION_TOKEN`: Notion integration's API token.
   - `NOTION_PARENT_PAGE_ID`: ID of the shared Notion parent page.
4. Push to GitHub. The workflow creates one Notion child page per push (short
   SHA, commit link, changed-file list, status). It adds **metrics and actual
   uploaded plots only when `experiments/latest.json` changed in that push**.
5. Inspect **Actions → PICART Push to Notion**. A green run and a resulting
   Notion subpage, including images, are required to confirm it actually works.

Plots are uploaded to Notion with the file-upload API, so they can display
inside Notion even when the GitHub repository is private. The file paths and
commit URL are also recorded for provenance.

The workflow does not run experiments itself, and it cannot solicit comments
on every push. Add your human interpretation to the experiment record before
pushing. A code-only push creates a code-change log with no stale experiment
numbers or old plots.

## Known source-equation ambiguity

The proposal's Eq. (11) appears algebraically inconsistent with Eq. (10).
Also, the `phi` sign in printed Algorithm 2 appears reversed relative to
Eq. (10)'s definition as a derivative target. The demonstration uses a
**positive** `phi=rate*parent_prediction` for the 1D ODE `f'=rate*f`;
it follows Algorithm 2's candidate-split loop but documents this proposed
correction. Verify both issues before treating this as a faithful reproduction.

## One methodological caution

For an `exp(rate*x)` function, the ODE `f'=rate*f` makes the desired slope
computable from a predicted function level. But using the parent response mean
to estimate `f` at the node midpoint is an *approximation*, not the true
midpoint value for nonlinear f. It is a testable modeling assumption.
