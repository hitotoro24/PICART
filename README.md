# PICART — Physics-Informed Regression Trees

My first PhD research project, with Dr. Nathan Wycoff (UMass Amherst), on teaching
**decision trees** to respect the **laws of physics**.

## The idea in plain words

Many prediction tools learn only from examples. Give them noisy measurements
and they will happily learn the noise, too. In science and engineering, though,
we often *already know* a rule the data must follow — for example, "this
quantity grows at a rate proportional to its size" (exponential growth), or
"heat spreads from hot to cold at a known speed". Such rules are written as
**differential equations**: equations about how fast something changes.

*Physics-informed machine learning* uses those rules as a second teacher. The
model is rewarded both for matching the data and for obeying the rule, so it
stays sensible even where data is scarce or noisy. This has worked well for
neural networks, but neural networks are hard to interpret.

**Decision trees** are among the most interpretable models: they split the data
with simple yes/no questions ("is x below 0.7?") and predict a constant value in
each resulting box. The catch is that a tree's prediction is a staircase — it is
flat almost everywhere, so its slope (rate of change) is either zero or
undefined. That makes it seem impossible to ask a tree to obey a rule about
rates of change.

The proposal behind this project gets around that with a **tree-based
derivative approximation**: the slope inside a box is estimated from the jump
between its two child boxes. With that estimate, the physics rule can be added
to the cost the tree minimises when choosing each split. The resulting method is
called **PICART** (Physics-Informed CART).

## What I'm working on now

Still have questions about the first-order PICART formulas. Right now I'm also
trying to get second derivatives from the first-derivative estimates in
*Regression Trees Know Calculus*. The current tests are on quadratics, first
with exact regional means and then with a fitted CART. Haven't worked out the
general smooth-function case or added a physics penalty yet.

## Progress log

Newest first. Each entry links to where the details live.

| Week | What happened | Where |
|---|---|---|
| 2026-W44 (Oct 26–Nov 1) | Worked through effective locations for tree-based first derivative estimates, then used pairs of nodes to try to recover Hessians in 2D and D dimensions. Checked exact box means against an actual fitted CART. Still need a good pair selection rule and general C² theory. | [`weeks/2026-W44/`](weeks/2026-W44/) |
| 2026-W43 (Oct 19–25) | Went through the four-grandchild 1D/2D stencils, unequal splits, mixed derivatives, and noise / curvature checks. | [`weeks/2026-W43/`](weeks/2026-W43/) |
| 2026-W42 (Oct 12–18) | Repository set up. Every equation of the proposal transcribed with notes on notational inconsistencies. A small one-split prototype of first-order PICART built and tested. Three issues found in the proposal's first-order algorithm (see the prototype's progress notes). Began studying the general first-order case. | [`weeks/2026-W42/`](weeks/2026-W42/), [`prototypes/first_order_stump/PROGRESS.md`](prototypes/first_order_stump/PROGRESS.md) |

## How this repository is organised

```
PICART/
├── README.md                   ← you are here: the big picture and overall progress
├── weeks/                      ← one folder per week: research notebook + supporting code and results
│   └── 2026-W42/
├── prototypes/                 ← small, self-contained experiments (side studies, not the main goal)
│   └── first_order_stump/      ← one-split PICART on exponential data; has its own PROGRESS.md
├── tests/                      ← checks for the shared tools
├── tools/                      ← automation (e.g. logging each push to Notion)
└── requirements.txt            ← Python packages needed
```

## Running things

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m unittest discover -s prototypes/first_order_stump/tests -v
```

Each week's notebook opens in Jupyter. Each prototype explains how to run it in
its own `PROGRESS.md`.

## Sources

- N. Wycoff, *Physics-Informed Regression Trees*, project description.
- N. Wycoff, *Regression Trees Know Calculus*, arXiv:2405.13846. Introduces the tree-based
  derivative approximation PICART builds on.
