"""Tree-based second-derivative estimation for a 3D quadratic target.

Notation follows docs/claude/*.tex and Wycoff (2024):
    f(x) = c + b^T x + 1/2 x^T Q x,   Q = Q^T,   Hessian = Q.
A box (cell) is C = prod_p [lo_p, hi_p]; mu_C is the mean of f over C.
"""
from .quadratic import Quadratic
from .stencils import (gradient_estimate, diag_three_cell, diag_four_cell,
                       mixed_four_cell, cells_diag_three, cells_diag_four,
                       cells_mixed_four)
from .tree_extract import (extract_nodes, split_table, midpoint_deviation,
                           find_diag_configs, find_mixed_configs, assemble)
