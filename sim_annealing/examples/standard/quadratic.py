"""
Created on Wed Aug  1 13:35:57 2018

@author: jordy
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
import sim_annealing as sa
from MyUtils.printing import title

# =============================================================================
### Simple quadratic function a^2x + bx + c
# =============================================================================


def quadratic(x, param):
    """
    Computes simple quadratic function evaluation
    """
    a, b, c = param
    # f is written to take an array as inputvalue, for consistency reasons
    # with examples with multiple inputs
    x = x[0]
    return a * x ** 2 + b * x + c


# solution is -2/3, with a=3, b=4, c=7
param = np.array([3.0, 4.0, 7.0])

# x0, needs to be an array (which is cumbersome in this simple example)
x0 = np.array([100.0])

# testing quadratic function
actual_solution = np.array([-2 / 3])
actual_minimum = quadratic(actual_solution, param)

# Start minimisation with annealing
quadr = sa.standard_annealer.Annealer(x0, quadratic, param, None, None)
quadr.max_updates = 200
quadr_results = quadr.anneal()

title("QUADRATIC FUNCTION")
quadr_results()
quadr_results.comparison_known_param(actual_solution, actual_minimum)
