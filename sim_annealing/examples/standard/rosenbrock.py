"""
Created on Wed Aug  1 13:35:57 2018

@author: jordy

Test the standard simulated annealing class on the Rosenbrock minimisation 
problem.
Gives an example on how to use the schedule method
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
from sim_annealing.standard_annealer import Annealer
from MyUtils.printing import title

# =============================================================================
### Rosenbrock
# =============================================================================
# function to minimise


def Rosenbrock(x):
    """
    evaluates the rosenbrock function with the elements of the given array
    """
    n = len(x)
    f = 0
    for k in range(0, n - 1):
        f += 100 * (x[k + 1] - x[k] ** 2) ** 2 + (1 - x[k]) ** 2

    return f


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
actual_minimum = Rosenbrock(actual_solution)

for i, ix in enumerate(x0):
    Ros_2d = Annealer(ix, Rosenbrock, None, None, None)
    Ros_2d.max_updates = 1000
    schedules = Ros_2d.schedule()
    Ros_2d.set_schedule(schedules[1])

    Ros_2d_results = Ros_2d.anneal()

    title("ROSENBROCK 2D")
    Ros_2d_results()
    Ros_2d_results.comparison_known_param(actual_solution, actual_minimum, 10e-3)

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
actual_minimum = Rosenbrock(actual_solution)

for i, ix in enumerate(x0):
    Ros_4d = Annealer(ix, Rosenbrock, None, None, None)
    Ros_4d.max_updates = 1000
    schedules = Ros_4d.schedule()
    Ros_4d.set_schedule(schedules[1])

    Ros_4d_results = Ros_4d.anneal(False)  # no intermediate prints

    title("ROSENBROCK 4D")
    Ros_4d_results()
    Ros_4d_results.comparison_known_param(actual_solution, actual_minimum, 10e-3)

"""
One should notice that it relatively takes quite some time.
However, this set-up guarantees finding the optimal solution.
It can be done faster, but in those cases, it could be that the
optimum is not found
"""
