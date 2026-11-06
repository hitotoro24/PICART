# Week 45 | IEEE PES: 3D quadratic Hessian, controlled stencils vs. a real CART

Continues W44. Same target, a quadratic `f(x) = c + b^T x + 1/2 x^T Q x`, now with a
fixed 3x3 `Q`, and the question split in two: does the finite-difference construction
on regional means work at all, and does an ordinary CART tree ever produce the
regions it needs?

Provenance: the code, tests and experiment runs in this folder were implemented with
Claude Code from a written brief (aligned, equally spaced regions, regional means;
stop at any unresolved mathematical choice). The derivations and the open questions
are mine. Full findings: [`report_3d_hessian.md`](report_3d_hessian.md).

## What is here

```
hessian_3d_2026-W45.ipynb   2D then 3D: Known Q / Estimated Hessian / error, four cases each
tree_hessian/               package: exact box means, stencils, CART extraction,
                            uneven-split (effective-centre) solver, 26 pytest tests
experiments/                run_controlled.py, run_cart.py, run_uneven.py + saved outputs
report_3d_hessian.md        research summary: recovered entries, assumptions, limits, open choices
```

Run from this folder:

```bash
python -m pytest -q tree_hessian/tests
python experiments/run_controlled.py
python experiments/run_cart.py
python experiments/run_uneven.py
```

## What I ran

1. **Aligned, equal cells, exact regional means.** Four-cell stencils for `Q_ii`
   (midpoint split, midpoint child splits) and `Q_ij` (2x2 block). All six entries
   exact, Frobenius error 1e-14. Violating equal widths gives a bias exactly linear in
   the threshold offset, scaled by the *diagonal* entry.
2. **Same cells, sample means.** Noiseless `y` still has Monte Carlo error from the
   variation of `f` inside a cell; observation noise of sd 0.1 barely changes it.
   RMSE falls roughly like `n^-1/2` and grows fast as cells shrink.
3. **sklearn CART on the same data.** Thresholds sit 5 to 9 % of the node width from
   the midpoint at every depth; the aligned configuration was accepted in zero
   subtrees in every tree. The shape "node splits `x_i`, both children split `x_i`"
   never appeared at depth 3 to 7, so diagonal entries are unreachable that way.
4. **Effective-centre extension (the W44 construction) on the CART partition.** With
   exact box means on the fitted boxes, all six entries exact at depth 5 and 7, rank 3
   in every row. With the tree's own leaf means, Frobenius error 0.17 to 0.35 at
   N = 20 000 and about 0.1 at N = 200 000; not monotone in depth. My own
   `cart_regressor.py` gives the identical splits to sklearn.

## Things I haven't resolved

- Pair weighting: all pairs equal, so sparse deep cells dominate at depth 10.
- Pooling all nodes splitting `x_s` over the whole domain is right only for a quadratic.
- Identifiability needs at least four nodes splitting a variable with affinely spanning
  effective centres; observed, not yet stated as a lemma.
- Two stencil facts observed numerically but not derived: the mixed stencil stays exact
  for any outer split if the inner thresholds coincide; the diagonal stencil stays exact
  for an off-centre outer split if each child splits at its own midpoint.
- Root extent still comes from the sampling design, not the data.
