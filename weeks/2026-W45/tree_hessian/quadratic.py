import numpy as np


class Quadratic:
    """f(x) = c + b^T x + 1/2 x^T Q x with symmetric Q.  nabla^2 f == Q."""

    def __init__(self, Q, b, c=0.0):
        """Store Q, b, c after checking Q is symmetric; d is the dimension."""
        Q = np.asarray(Q, dtype=float)
        if not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric")
        self.Q, self.b, self.c = Q, np.asarray(b, dtype=float), float(c)
        self.d = Q.shape[0]

    def __call__(self, X):
        """Evaluate f at each row of X (shape (n, d) or (d,)); returns shape (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # einsum rather than X @ b: the BLAS matmul path on this macOS numpy build
        # raises spurious floating-point warnings for long inputs.
        return (self.c + np.einsum("ni,i->n", X, self.b)
                + 0.5 * np.einsum("ni,ij,nj->n", X, self.Q, X))

    # ---- exact regional average -------------------------------------------
    # Two-point Gauss-Legendre per axis (nodes at +-1/sqrt(3) of the half-width,
    # equal weights) integrates any polynomial of degree <= 3 in each variable
    # exactly, so for a quadratic the box mean below has only rounding error.
    _GL = np.array([-1.0, 1.0]) / np.sqrt(3.0)

    def box_mean_exact(self, lo, hi):
        """Exact mean of f over the box [lo, hi] under the uniform measure.
        Uses the tensor two-point Gauss-Legendre rule, which is exact for
        a quadratic, so the only error is floating-point rounding."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        grids = np.meshgrid(*[mid[p] + half[p] * self._GL for p in range(self.d)],
                            indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=1)
        return float(self(pts).mean())

    # ---- sampled regional average ------------------------------------------
    def sample_uniform(self, lo, hi, n, rng, noise_sd=0.0):
        """n points uniform on the box, y = f(x) + N(0, noise_sd^2)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        X = lo + rng.random((n, self.d)) * (hi - lo)
        y = self(X)
        if noise_sd > 0:
            y = y + noise_sd * rng.standard_normal(n)
        return X, y


def box_mean_sample(X, y, lo, hi):
    """Sample mean of y over points with lo < x <= hi (sklearn's convention:
    a point goes left when x <= threshold). Returns (mean, count); mean is NaN
    for an empty cell."""
    mask = np.all((X > lo) & (X <= hi), axis=1)
    k = int(mask.sum())
    return (float(y[mask].mean()) if k else np.nan), k
