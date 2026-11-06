"""Controlled test of second-order finite differences on regional means for
the 3D quadratic f(x) = c + b^T x + 1/2 x^T Q x on [0,1]^3.

Part A: exact regional averaging  -> deterministic approximation error.
Part B: exact averaging with assumption (A2) violated -> size of the bias.
Part C: uniform sampling          -> sampling (Monte Carlo) error.
Run:  .venv/bin/python experiments/run_controlled.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from tree_hessian import (Quadratic, diag_three_cell, diag_four_cell, mixed_four_cell,
                          cells_diag_three, cells_diag_four, cells_mixed_four)
from tree_hessian.quadratic import box_mean_sample

Q = np.array([[4.0, 1.5, 0.5], [1.5, 3.0, -1.0], [0.5, -1.0, 2.0]])
b = np.array([-1.0, 2.0, 0.5]); c = 0.7
f = Quadratic(Q, b, c)
LO, HI = np.zeros(3), np.ones(3)
PAIRS = [(0, 1), (0, 2), (1, 2)]
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("Q =\n", Q, "\nb =", b, " c =", c, "\n")


def estimate_all(mean_fn, lo, hi):
    """All six entries from four-cell stencils on the box [lo, hi]."""
    H = np.full((3, 3), np.nan)
    for i in range(3):
        mu = [mean_fn(*cell) for cell in cells_diag_four(lo, hi, i)]
        H[i, i] = diag_four_cell(*mu, lo[i], hi[i])
    for i, j in PAIRS:
        mu = [mean_fn(*cell) for cell in cells_mixed_four(lo, hi, i, j)]
        H[i, j] = H[j, i] = mixed_four_cell(*mu, lo[i], hi[i], lo[j], hi[j])
    return H


# ---------------------------------------------------------------- Part A
print("=" * 78)
print("PART A: exact regional averaging (deterministic approximation error)")
print("=" * 78)
H = estimate_all(f.box_mean_exact, LO, HI)
print("box [0,1]^3, four-cell stencils:\n", H)
print("entrywise |error|:\n", np.abs(H - Q))
print("Frobenius ||H - Q|| = %.2e" % np.linalg.norm(H - Q))
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):          # random sub-boxes of [0,1]^3, random cell widths
    lo = rng.uniform(0, 0.5, 3); hi = lo + rng.uniform(0.05, 0.5, 3)
    worst = max(worst, np.abs(estimate_all(f.box_mean_exact, lo, hi) - Q).max())
    for i in range(3):
        h = rng.uniform(0.01, 0.2)
        mu = [f.box_mean_exact(*cl) for cl in cells_diag_three(lo, hi, i, h, rng.uniform(0.3, 0.7))]
        worst = max(worst, abs(diag_three_cell(*mu, h) - Q[i, i]))
print("max |error| over 200 random boxes/widths, three- and four-cell stencils: %.2e" % worst)

# ---------------------------------------------------------------- Part B
print("\n" + "=" * 78)
print("PART B: exact averaging, assumption (A2) violated (bias, not sampling)")
print("=" * 78)
print("B1  diag_four_cell on x0, left child split at 0.25 + delta (parent [0,1]):")
print("    delta    estimate   error    error/delta")
for delta in (0.0, 0.01, 0.05, 0.10, 0.20):
    mu = [f.box_mean_exact(*cl) for cl in cells_diag_four(LO, HI, 0, tau_L=0.25 + delta)]
    e = diag_four_cell(*mu, 0.0, 1.0)
    print("   %5.2f   %8.4f   %8.4f   %s" % (delta, e, e - Q[0, 0], "%.3f" % ((e - Q[0, 0]) / delta) if delta else "-"))
print("B2  mixed_four_cell (0,1): x0-thresholds tau_B = 0.5-delta, tau_T = 0.5+delta:")
print("    delta    estimate   error    error/delta   (Q00 = %.1f)" % Q[0, 0])
for delta in (0.0, 0.01, 0.05, 0.10, 0.20):
    mu = [f.box_mean_exact(*cl) for cl in cells_mixed_four(LO, HI, 0, 1, tau_B=0.5 - delta, tau_T=0.5 + delta)]
    e = mixed_four_cell(*mu, 0, 1, 0, 1)
    print("   %5.2f   %8.4f   %8.4f   %s" % (delta, e, e - Q[0, 1], "%.3f" % ((e - Q[0, 1]) / delta) if delta else "-"))
print("B3  OBSERVATION (not derived): mixed_four_cell (0,1) with outer x1-split off-centre,")
print("    x0-thresholds both at the midpoint:")
for tau_j in (0.2, 0.35, 0.5, 0.8):
    mu = [f.box_mean_exact(*cl) for cl in cells_mixed_four(LO, HI, 0, 1, tau_j=tau_j)]
    print("    tau_1 = %.2f   estimate %.6f   error %.1e" % (tau_j, mixed_four_cell(*mu, 0, 1, 0, 1), mixed_four_cell(*mu, 0, 1, 0, 1) - Q[0, 1]))
print("B4  OBSERVATION (not derived): diag_four_cell x0 with outer split off-centre, child")
print("    splits at their own midpoints, formula uses (gamma_R-gamma_L)/((w_L+w_R)/2):")
for tau0 in (0.3, 0.5, 0.7):
    cl = cells_diag_four(LO, HI, 0, tau_node=tau0)
    mu = [f.box_mean_exact(*x) for x in cl]
    gL = (mu[1] - mu[0]) / (tau0 / 2); gR = (mu[3] - mu[2]) / ((1 - tau0) / 2)
    print("    tau_0 = %.2f   estimate %.6f   error %.1e" % (tau0, (gR - gL) / 0.5, (gR - gL) / 0.5 - Q[0, 0]))

# ---------------------------------------------------------------- Part C
print("\n" + "=" * 78)
print("PART C: uniform sampling on [0,1]^3, sample means in the cells (sampling error)")
print("=" * 78)
print("Stencil box = [0,w]^3 (cells have width w/4 (diag) or w/2 (mixed) along the")
print("differenced axes).  RMSE over 10 seeds; 'empty' = seeds with an empty cell.")
SEEDS = 10
for noise_sd in (0.0, 0.1):
    print("\n--- observation noise sd = %.1f ---" % noise_sd)
    print("   w      n     RMSE Q00  RMSE Q11  RMSE Q22  RMSE Q01  RMSE Q02  RMSE Q12  max|err|  Frob RMSE  empty")
    for w in (1.0, 0.5, 0.25):
        lo, hi = LO, LO + w
        for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
            errs, empty = [], 0
            for seed in range(SEEDS):
                X, y = f.sample_uniform(LO, HI, n, np.random.default_rng(seed), noise_sd)
                Hs = estimate_all(lambda a, b_: box_mean_sample(X, y, a, b_)[0], lo, hi)
                if np.isnan(Hs).any():
                    empty += 1; continue
                errs.append(Hs - Q)
            if not errs:
                print("  %4.2f  %7d   (all seeds had an empty cell)" % (w, n)); continue
            E = np.array(errs)
            rmse = np.sqrt((E ** 2).mean(axis=0))
            frob = np.sqrt((np.linalg.norm(E, axis=(1, 2)) ** 2).mean())
            print("  %4.2f  %7d   %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f   %d/%d"
                  % (w, n, rmse[0, 0], rmse[1, 1], rmse[2, 2], rmse[0, 1], rmse[0, 2], rmse[1, 2],
                     np.abs(E).max(), frob, empty, SEEDS))
