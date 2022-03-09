"""
Created on Thu Aug  2 10:04:50 2018

@author: jordy

This script shows the example of the the Maximum likelihood parameter 
estimation of an AR(1) process. Also, the example illustrates the use of njit
functions together with jitclass.
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np

from sim_annealing.jitclass_annealer import Annealer
from MyUtils.printing import title
from MyUtils.simulation import ornstein_uhlenbeck as ous
from MyUtils.estimation import ornstein_uhlenbeck as oue
from MyUtils.estimation.normal import ci
from MyUtils.estimation.general import check_within_ci
from MyUtils.estimation.general import comparison_actual_solution

from collections import OrderedDict
from numba import types, jitclass, njit
import matplotlib.pyplot as plt


# =============================================================================
### AR(1)
# x_{t+1} = a + b * x_t + eps_{t+1}, with eps_{t+1} ~ N(0, sigma_e^2)
# =============================================================================
# simulate dataset
# actual parameters
kappa = 0.076278547919064900
theta = 0.0
sigma = 1.0
nbr_timepoints = 5000
nbr_simul = 1
starting_value = 0.5

ou_param = np.array([kappa, theta, sigma])

data = ous.univariate(
    starting_value, kappa, theta, sigma, nbr_timepoints, nbr_simul
).reshape(-1)
plt.plot(data)

# GAUSSIAN AR(1) Process
@njit
def loglikelihood_gaussian_AR1(param, t):
    """
    Computes the loglikelihood for an AR(1) process, given data and parameters
    """
    y_t = t[1:]
    y_tm1 = t[:-1]

    a, b, sigma_e = param
    T = len(t)

    # conditional distribution of all observations,
    # given the previous one and assuming a known y_1
    ll_t = -(T - 1) / 2 * np.log(2 * np.pi * sigma_e ** 2) - 1 / 2 * np.sum(
        (y_t - a - b * y_tm1) ** 2 / sigma_e ** 2
    )

    return ll_t


# Annealer
ar1_spec = OrderedDict()
ar1_spec["data"] = types.Array(types.float64, 1, "A")


@jitclass(ar1_spec)
class GaussianAR1(object):
    def __init__(self, data):
        self.data = data

    def energy(self, param):
        return -loglikelihood_gaussian_AR1(param, self.data)


@jitclass([])
class AR1Accept(object):
    def __init__(self):
        pass

    def check(self, state):
        for i in state:
            if abs(i) > 10e22:
                return False
        if state[2] < 0:
            return False
        return True


# actual values
# transform ou parameters into AR(1) regression param
actual_solution = oue.lr_param_from_ou_param(ou_param)
actual_minimum = GaussianAR1(data).energy(actual_solution)

x0 = np.ones_like(actual_solution)  # initial guess

ar1 = Annealer(GaussianAR1, data, AR1Accept)
results = ar1.anneal(x0, max_updates=500)

title("AR(1)")
ar1.report_results()
ar1.comparison_actual_solution(actual_solution, actual_minimum, 5)

title("OU-parameters")
ou_param_est = oue.ou_param_from_lr_param(results.state)
print("OU estimate", ou_param_est)
print("OU actual", ou_param)
comparison_actual_solution(ou_param_est, ou_param, 0.2)
