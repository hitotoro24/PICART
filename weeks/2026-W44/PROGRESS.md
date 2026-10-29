# Week 44 | Hessian from tree derivative estimates

I spent most of this week trying to go from the first derivative estimator in *Regression Trees Know Calculus* to a Hessian. Not the physics penalty yet. Just figuring out if I can recover second derivatives from the tree in the first place.

What threw me off was the location issue. gamma_i gives a first derivative estimate from two child means, but where does it actually live? For a quadratic with **uniform box averages**, I worked out an effective location z_i. The split direction has the (l + tau + u)/3 term, while the other coordinates use the midpoint. It's not just the center of the node.

From there the idea is pretty simple, at least algebraically:
- Two nodes A and B split along the same coordinate sigma.
- Calculate gamma_A and gamma_B from their child means.
- Their effective locations give v_AB = z_B - z_A. That's a displacement, not another derivative.
- For a quadratic, gamma_B - gamma_A = sum_j Q[sigma,j] * v_AB[j].
- Stack those equations and solve for the symmetric Hessian Q. Check rank first. More equations alone don't mean shit if they're all giving the same information.

## What I ran

See [the notebook](tree_hessian_2D_to_D.ipynb).

1. **2D, no actual tree yet.** Used rectangular regions and exact uniform box means to see if the algebra works. It recovers the known Q to numerical precision.
2. **D dimensions, D=4 in this run.** Same construction. Ten unknown entries, no new type of node pair needed. Also recovers Q to numerical precision.
3. **Actual CART.** Fitted sklearn DecisionTreeRegressor on sampled quadratic data, extracted internal-node regions, split coordinates and thresholds. Then compared Hessian recovery using the empirical child means versus exact box means on the *same* fitted partitions. Empirical error was nonzero (about 0.22 in 2D and 0.20 in 4D); the exact-box-means version recovers Q to numerical precision.

## Things I haven't resolved

- This is only an exact identity for quadratic functions with exact uniform box averages. CART's averages aren't exact integrals.
- Pair selection is still crude: same sigma, then a few nearest effective locations. It might be ill-conditioned, and local pairs can amplify the error in gamma.
- I haven't shown consistency for a general C^2 function. Need to look at the Taylor error and the geometry of nearby nodes rather than treating this numerical check as a proof.
- This is Hessian estimation from a tree. Not yet a physics-informed CART algorithm.

Next thing I want to understand: how to choose useful node pairs and what assumptions would make the general smooth case work. The linear system itself isn't the difficult part.
