"""
Created on Wed Aug  1 14:37:47 2018

@author: jordy

Test the jitclass simulated annealing class on the Rosenbrock minimisation 
problem.

shows use of schedule
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np

from numba import jitclass
from sim_annealing.jitclass_annealer import Annealer
from MyUtils.printing import title

# =============================================================================
### Rosenbrock
# =============================================================================
# functionclass to minimise
@jitclass([])
class Rosenbrock(object):
    def __init__(self):
        pass

    def energy(self, state):
        n = len(state)
        f = 0
        for k in range(0, n - 1):
            f += 100 * (state[k + 1] - state[k] ** 2) ** 2 + (1 - state[k]) ** 2
        return f


@jitclass([])
class Ros_accept(object):
    def __init__(self):
        pass

    def check(self, state):
        respons = True
        for i in state:
            if abs(i) > 10e22:
                respons = False
                break

        return respons


# 2-D
# different starting values
x0 = np.array(
    [
        [1001.0, 1001.0],
        [1001.0, -999.0],
        [-999.0, -999.0],
        [-999.0, 1001.0],
        [1443.0, 1.0],
        [1.0, 1443.0],
        [1.2, 1.0],
    ]
)

actual_solution = np.array([1.0, 1.0])
actual_minimum = Rosenbrock().energy(actual_solution)

for i, ix in enumerate(x0):
    Ros_2d = Annealer(Rosenbrock, None, Ros_accept)
    results = Ros_2d.anneal(ix, max_updates=500, auto_schedule=True)

    title("ROSENBROCK 2D")
    Ros_2d.report_results()
    Ros_2d.comparison_actual_solution(actual_solution, 0, 0.01)

# 4-D
# different starting values
x0 = np.array(
    [
        [101.0, 101.0, 101.0, 101.0],
        [101.0, 101.0, 101.0, -99.0],
        [101.0, 101.0, -99.0, -99.0],
        [101.0, -99.0, -99.0, -99.0],
        [-99.0, -99.0, -99.0, -99.0],
        [-99.0, 101.0, -99.0, 101.0],
        [101.0, -99.0, 101.0, -99.0],
        [201.0, 0.0, 0.0, 0.0],
        [1.0, 201.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 201.0],
    ]
)

actual_solution = np.ones(4)
actual_minimum = Rosenbrock().energy(actual_solution)

for i, ix in enumerate(x0):
    Ros_4d = Annealer(Rosenbrock, None, Ros_accept)
    results = Ros_4d.anneal(ix, max_updates=500, auto_schedule=True)

    title("ROSENBROCK 4D")
    Ros_4d.report_results()
    Ros_4d.comparison_actual_solution(actual_solution, 0, 0.01)

