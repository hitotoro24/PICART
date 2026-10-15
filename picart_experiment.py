"""Small first-order PICART experiment, grounded in proposal Algorithm 2.

Only a one-dimensional depth-one tree is implemented. This is deliberately NOT a
complete recursive PICART or an exact reproduction of the proposal's Figure 2.

Run: python picart_experiment.py --seed 17 --n 120 --sigma 1.5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def synthetic_exponential(n: int, sigma: float, seed: int, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f(x)=exp(rate*x), f'(x)=rate*f(x), x in [0,1]."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    truth = np.exp(rate * x)
    y = truth + rng.normal(0, sigma, size=n)
    return x, y, truth


def solve_child_predictions(
    n_left: int, n_right: int, mean_left: float, mean_right: float,
    width: float, target_derivative: float, lambda_tilde: float,
) -> tuple[float, float]:
    """Solve the correct 2x2 FOCs of proposal Eq. (10); cf. printed Eq. (11).

    Under the fixed target-derivative convention, this is a strictly convex
    quadratic when both child counts are positive and width > 0.
    """
    if min(n_left, n_right) < 1 or width <= 0 or lambda_tilde < 0:
        raise ValueError("Need two nonempty children, positive width, nonnegative penalty")
    beta = 4.0 * lambda_tilde / width**2
    a = np.array([[n_left + beta, -beta], [-beta, n_right + beta]], dtype=float)
    b = np.array([
        n_left * mean_left - 2 * lambda_tilde * target_derivative / width,
        n_right * mean_right + 2 * lambda_tilde * target_derivative / width,
    ], dtype=float)
    result = np.linalg.solve(a, b)
    return float(result[0]), float(result[1])


def candidate_thresholds(x: np.ndarray) -> np.ndarray:
    """Midpoints of successive distinct predictor values, as in greedy CART."""
    unique = np.unique(x)
    return (unique[:-1] + unique[1:]) / 2.0


def fit_picart_stump(x: np.ndarray, y: np.ndarray, lambda_d: float, rate: float) -> dict:
    """One-dimensional depth-one specialization of proposal Algorithm 2.

    Algorithm-2 flow: for node -> feature p -> target phi -> threshold tau
    -> minimize Eq. (10) in child predictions -> retain best split.

    For 1D f'(x)=rate*f(x), target phi = rate*(existing parent prediction).
    Following the proposal, the parent prediction is held FIXED during candidate
    split search; changing this modeling choice is a separate experiment.
    CAUTION: the sign in Algorithm 2's printed phi line seems inconsistent with
    the role of phi as the derivative *target* in Eq. (10). Here, for the 1D
    physical law f'=rate*f, we set phi=+rate*parent_prediction. We preserve
    the pseudocode's control flow, but do NOT silently accept its suspect sign.

    Full split score MUST include within-child SSE constants omitted in Eq. (10).
    """
    x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2 or lambda_d < 0:
        raise ValueError("Bad data shapes, sample size, or lambda_d")
    low, high = 0.0, 1.0  # known simulation/domain cell bounds, not sample extrema
    width = high - low
    volume = width  # one-dimensional node volume
    lam_tilde = lambda_d * volume
    parent_prediction = float(y.mean())
    phi = rate * parent_prediction
    best = None
    for threshold in candidate_thresholds(x):
        mask = x < threshold
        nl, nr = int(mask.sum()), int((~mask).sum())
        if not nl or not nr:
            continue
        ml, mr = float(y[mask].mean()), float(y[~mask].mean())
        mu_l, mu_r = solve_child_predictions(nl, nr, ml, mr, width, phi, lam_tilde)
        rss = float(np.sum((y[mask] - mu_l)**2) + np.sum((y[~mask] - mu_r)**2))
        residual = 2.0 * (mu_r - mu_l) / width - phi
        objective = rss + lam_tilde * residual**2
        if best is None or objective < best["objective"]:
            best = dict(threshold=float(threshold), n_left=nl, n_right=nr,
                        mean_left=ml, mean_right=mr, mu_left=mu_l, mu_right=mu_r,
                        rss=rss, physics_residual=float(residual), objective=float(objective),
                        phi=phi, lambda_d=float(lambda_d), lambda_tilde=float(lam_tilde))
    if best is None:
        raise ValueError("No feasible split")
    return best


def predict_stump(model: dict, x: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(x) < model["threshold"], model["mu_left"], model["mu_right"])


def run_experiment(args: argparse.Namespace) -> Path:
    if args.n < 4 or args.sigma < 0 or args.rate <= 0:
        raise ValueError("n >= 4, sigma >= 0, rate > 0 required")
    x, y, _ = synthetic_exponential(args.n, args.sigma, args.seed, args.rate)
    grid = np.linspace(0.0, 1.0, 500)
    f_grid = np.exp(args.rate * grid)
    cart = DecisionTreeRegressor(max_depth=1, random_state=args.seed)
    cart.fit(x.reshape(-1, 1), y)
    cart_grid = cart.predict(grid.reshape(-1, 1))
    cart_rmse = float(np.sqrt(np.mean((cart_grid - f_grid)**2)))
    levels = [float(z) for z in args.lambdas.split(",")]
    if not levels or any(z < 0 for z in levels):
        raise ValueError("lambdas must be nonnegative comma-separated numbers")
    models = [fit_picart_stump(x, y, z, args.rate) for z in levels]
    for model in models:
        fit = predict_stump(model, grid)
        model["test_rmse_vs_truth"] = float(np.sqrt(np.mean((fit-f_grid)**2)))
        model["physics_residual_abs"] = abs(model["physics_residual"])
        model["prediction_shift_left"] = model["mu_left"] - model["mean_left"]
        model["prediction_shift_right"] = model["mu_right"] - model["mean_right"]
    out = Path("experiments")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=12, alpha=0.45, label="Noisy observations")
    ax.plot(grid, f_grid, linewidth=2, label="True exp(rate·x)")
    ax.plot(grid, cart_grid, linewidth=2, label="sklearn CART (depth 1)")
    for model in models:
        if model["lambda_d"] in (levels[0], levels[-1]):
            ax.plot(grid, predict_stump(model, grid), label=f"PICART λ={model['lambda_d']:g}")
    ax.legend(fontsize=8)
    ax.set(xlabel="x", ylabel="y", title="First-order PICART stump: illustrative experiment")
    fig.tight_layout()
    fig.savefig(out / "fit.png", dpi=150)
    plt.close(fig)

    # In addition to optimal split behavior, hold CART's threshold fixed to
    # isolate the influence of penalty weight on child predictions.
    cart_tau = float(cart.tree_.threshold[0])
    mask = x < cart_tau
    fixed = []
    phi = args.rate * float(y.mean())
    for z in levels:
        mu_l, mu_r = solve_child_predictions(int(mask.sum()), int((~mask).sum()),
                        float(y[mask].mean()), float(y[~mask].mean()), 1.0, phi, z)
        fixed.append({"lambda_d": z, "mu_left": mu_l, "mu_right": mu_r,
                      "residual_abs": abs(2*(mu_r-mu_l)-phi)})
    fig, (leftax, rightax) = plt.subplots(1, 2, figsize=(9, 4))
    leftax.plot(levels, [z["mu_left"] for z in fixed], "o-", label="Left prediction")
    leftax.plot(levels, [z["mu_right"] for z in fixed], "o-", label="Right prediction")
    leftax.set_xscale("symlog", linthresh=0.01)
    leftax.set(xlabel="λd (symmetric log scale)", ylabel="Prediction", title="Fixed CART split: child predictions")
    leftax.legend(fontsize=8)
    rightax.plot(levels, [z["residual_abs"] for z in fixed], "o-")
    rightax.set_xscale("symlog", linthresh=0.01)
    rightax.set(xlabel="λd (symmetric log scale)", ylabel="|physics residual|", title="Fixed split: residual")
    fig.tight_layout()
    fig.savefig(out / "lambda_sweep.png", dpi=150)
    plt.close(fig)

    results = {
        "experiment_id": datetime.now(timezone.utc).strftime("picart-%Y%m%dT%H%M%SZ"),
        "description": "First-order PICART stump using proposal Algorithm 2, compared with sklearn CART",
        "hypothesis": "As physics penalty increases, left prediction decreases and right increases (conditional, not universal).",
        "method": "One-dimensional exp(rate*x) with Gaussian noise; fixed and reoptimized candidate splits",
        "seed": args.seed, "n": args.n, "sigma": args.sigma, "rate": args.rate,
        "cart_rmse_vs_truth": cart_rmse,
        "cart_threshold": cart_tau,
        "picart_runs": models,
        "fixed_split_sweep": fixed,
        "figures": ["experiments/fit.png", "experiments/lambda_sweep.png"],
        "interpretation": "PENDING: write your observations before pushing results",
        "next_step": "PENDING: choose the next hypothesis or code change",
        "note": "Illustrative test only; not a verified replication of proposal Figure 2.",
    }
    (out / "latest.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out/'latest.json'}, fit.png, lambda_sweep.png")
    for model in models:
        print(f"lambda={model['lambda_d']:>7g} tau={model['threshold']:.3f} "
              f"mu=({model['mu_left']:.3f},{model['mu_right']:.3f}) "
              f"test_rmse={model['test_rmse_vs_truth']:.3f} "
              f"abs_physics_resid={model['physics_residual_abs']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--lambdas", default="0,0.01,0.1,1,10,100")
    run_experiment(ap.parse_args())


if __name__ == "__main__":
    main()
