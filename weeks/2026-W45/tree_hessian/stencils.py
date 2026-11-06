"""Second-order finite differences on regional means of aligned, equally
spaced cells.  Every formula here is a difference of two tree-style gradient
estimates (Wycoff 2024, eq. 3) divided by the distance between the regions
they are attached to.  Exactness for a quadratic requires:
  (A1) the cells are axis-aligned boxes sharing every extent except along the
       differenced axis/axes, and
  (A2) the cells have equal width along the differenced axis/axes.
No correction for uneven or misaligned splits is applied."""
import numpy as np


def gradient_estimate(mu_left, mu_right, lo_s, hi_s):
    """Wycoff (2024) eq. 3:  gamma = 2 (mu_r - mu_l) / (u_s - l_s),
    for a node with extent [lo_s, hi_s] along its split variable s."""
    return 2.0 * (mu_right - mu_left) / (hi_s - lo_s)


# ---- diagonal entry Q_ii -----------------------------------------------------
def diag_three_cell(mu_minus, mu_0, mu_plus, h):
    """Three adjacent cells of width h along axis i (classic stencil):
        Q_ii = (mu_plus - 2 mu_0 + mu_minus) / h^2.
    Equals (gamma_right - gamma_left)/h with gamma = (difference of adjacent
    means)/h, i.e. a second difference of first differences."""
    return (mu_plus - 2.0 * mu_0 + mu_minus) / h ** 2


def diag_four_cell(mu_LL, mu_LR, mu_RL, mu_RR, lo_i, hi_i):
    """Binary-tree form.  Node [lo_i, hi_i] split at its midpoint into L, R;
    each half split again at its own midpoint (four equal cells of width
    (hi_i-lo_i)/4).  gamma_L, gamma_R are eq.-3 estimates on L and R, whose
    centroids are (hi_i-lo_i)/2 apart:
        Q_ii = (gamma_R - gamma_L) / ((hi_i-lo_i)/2)
             = 8 (mu_RR - mu_RL - mu_LR + mu_LL) / (hi_i-lo_i)^2."""
    w = hi_i - lo_i
    gamma_L = gradient_estimate(mu_LL, mu_LR, lo_i, lo_i + w / 2)
    gamma_R = gradient_estimate(mu_RL, mu_RR, lo_i + w / 2, hi_i)
    return (gamma_R - gamma_L) / (w / 2)


# ---- mixed entry Q_ij ----------------------------------------------------------
def mixed_four_cell(mu_BL, mu_BR, mu_TL, mu_TR, lo_i, hi_i, lo_j, hi_j):
    """Node split at its midpoint along j into bottom B and top T; both halves
    split along i at the SAME threshold, the midpoint of [lo_i, hi_i].  Each
    half carries an eq.-3 estimate of d f / d x_i; their centroids are
    (hi_j-lo_j)/2 apart along j:
        Q_ij = (gamma_i(T) - gamma_i(B)) / ((hi_j-lo_j)/2)
             = 4 (mu_TR - mu_TL - mu_BR + mu_BL) / ((hi_i-lo_i)(hi_j-lo_j))."""
    gamma_B = gradient_estimate(mu_BL, mu_BR, lo_i, hi_i)
    gamma_T = gradient_estimate(mu_TL, mu_TR, lo_i, hi_i)
    return (gamma_T - gamma_B) / ((hi_j - lo_j) / 2)


# ---- cell constructors (return lists of (lo, hi) boxes) -------------------------
def _box(lo, hi, axis, a, b):
    """Copy of the box [lo, hi] with its extent along `axis` replaced by [a, b]."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    lo[axis], hi[axis] = a, b
    return lo, hi


def cells_diag_three(lo, hi, i, h, center):
    """Three cells of width h along axis i centred at `center`, full extent
    [lo, hi] on the other axes.  Order: minus, 0, plus."""
    edges = center + h * np.array([-1.5, -0.5, 0.5, 1.5])
    return [_box(lo, hi, i, edges[k], edges[k + 1]) for k in range(3)]


def cells_diag_four(lo, hi, i, tau_node=None, tau_L=None, tau_R=None):
    """Four cells along axis i from a node split (tau_node) and the two child
    splits (tau_L, tau_R).  Defaults are the midpoints = assumption (A2).
    Order: LL, LR, RL, RR."""
    a, b = lo[i], hi[i]
    t0 = (a + b) / 2 if tau_node is None else tau_node
    tL = (a + t0) / 2 if tau_L is None else tau_L
    tR = (t0 + b) / 2 if tau_R is None else tau_R
    return [_box(lo, hi, i, a, tL), _box(lo, hi, i, tL, t0),
            _box(lo, hi, i, t0, tR), _box(lo, hi, i, tR, b)]


def cells_mixed_four(lo, hi, i, j, tau_j=None, tau_B=None, tau_T=None):
    """2x2 cells in the (i, j) plane: split along j at tau_j into B (below)
    and T (above); B split along i at tau_B, T at tau_T.  Defaults are the
    midpoints = assumptions (A1)+(A2).  Order: BL, BR, TL, TR."""
    tj = (lo[j] + hi[j]) / 2 if tau_j is None else tau_j
    ti = (lo[i] + hi[i]) / 2
    tB = ti if tau_B is None else tau_B
    tT = ti if tau_T is None else tau_T
    B_lo, B_hi = _box(lo, hi, j, lo[j], tj)
    T_lo, T_hi = _box(lo, hi, j, tj, hi[j])
    return [_box(B_lo, B_hi, i, lo[i], tB), _box(B_lo, B_hi, i, tB, hi[i]),
            _box(T_lo, T_hi, i, lo[i], tT), _box(T_lo, T_hi, i, tT, hi[i])]
