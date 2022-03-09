"""
Created on Thu Aug  2 09:18:45 2018

@author: jordy

This script shows the example of the the Maximum likelihood parameter 
estimation of the Vasicek model, using jitclass and the jitclass annealer.
"""

# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
from sim_annealing.jitclass_annealer import Annealer
from MyUtils.printing import title
from MyUtils.simulation import ornstein_uhlenbeck as ous
from collections import OrderedDict
from numba import types, jitclass

# =============================================================================
### Vasicek
# =============================================================================
# simulate dataset
# actual parameters
kappa = 0.5
theta = 0
sigma = 0.2
nbr_timepoints = 500
nbr_simul = 1
starting_value = 0.5

data = ous.univariate(starting_value, kappa, theta, sigma, nbr_timepoints, nbr_simul)

# Annealing
# objective function
energy_class_spec = OrderedDict()
energy_class_spec["x0"] = types.Array(types.float64, 2, "A")
energy_class_spec["x1"] = types.Array(types.float64, 2, "A")


@jitclass(energy_class_spec)
class Vasicek(object):
    def __init__(self, data):
        self.x0 = data[:-1]
        self.x1 = data[1:]

    def energy(self, state):
        kappa = state[0]
        theta = state[1]
        sigma = state[2]
        de = 1

        output = (
            (2 * np.pi) ** (-1 / 2)
            * (sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * de))) ** (-1 / 2)
            * np.exp(
                -((self.x1 - theta - (self.x0 - theta) * np.exp(-kappa * de)) ** 2)
                / (2 * sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * de)))
            )
        )

        return -np.log(output).sum()


@jitclass([])
class AcceptRejectState(object):
    def __init__(self):
        pass

    def check(self, state):
        response = True
        if abs(state[0]) > 10e1:
            response = False
        elif abs(state[1]) > 10e1:
            response = False
        elif (abs(state[2]) > 10e1) or (state[2] < 0):
            response = False
        return response


# initial value + actual solutions
actual_solution = np.array([kappa, theta, sigma])
actual_minimum = Vasicek(data).energy(actual_solution)
x0 = np.ones_like(actual_solution)

# It is important to note that whenever real constrictions are put on the
# parameters, that the maximum stepsize (V_max) is adapted accordingly
vasicek = Annealer(Vasicek, data, AcceptRejectState)
res = vasicek.anneal(x0, max_updates=1000, V_max=np.ones_like(x0))

title("Vasicek")
vasicek.report_results()
vasicek.comparison_actual_solution(actual_solution, actual_minimum, 5)

"""
At first this might seem slow, however, speed per function evaluation 
rose drastically thanks to the use of jitclass
"""
