# =============================================================================
### Packages/Imports
# =============================================================================

"""
Created on Wed Aug  1 13:35:57 2018

@author: jordy

This script shows a simple example of the minimisation of a quadratic function
using jitclass and the jitclass annealer
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np

from numba import types
from numba.experimental import jitclass
from collections import OrderedDict
from sim_annealing.jitclass_annealer import Annealer
from sim_annealing._utils import title

# =============================================================================
### Simple quadratic function a^2x + bx + c
# =============================================================================
# jitclass needsyou to specify the types of attributes
quadr_spec = OrderedDict()
quadr_spec["a"] = types.float64
quadr_spec["b"] = types.float64
quadr_spec["c"] = types.float64

# object function in the form of a jitclass


@jitclass(quadr_spec)
class Quadr(object):
    def __init__(self, param):
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]

    def energy(self, state):
        return self.a * state[0] ** 2 + self.b * state[0] + self.c


# accept/reject must be written, but setting the domain large enough
# is like there are no restrictions at all
@jitclass([])
class Quadr_check(object):
    def __init__(self):
        pass

    def check(self, state):
        if abs(state[0]) > 10e500:
            return False
        else:
            return True


# solution is -2/3, with a=3, b=4, c=7
param = np.array([3.0, 4.0, 7.0])

# x0, needs to be an array (which is cumbersome in this simple example)
x_0 = np.array([100.0])

# testing quadratic function
actual_solution = np.array([-2 / 3])
actual_minimum = Quadr(param).energy(actual_solution)


# Start minimisation with annealing
quadr = Annealer(Quadr, param, Quadr_check)
quadr_results = quadr.anneal(x0=x_0, max_updates=200)

title("QUADRATIC FUNCTION")
quadr.report_results()
quadr.comparison_actual_solution(actual_solution, actual_minimum)

