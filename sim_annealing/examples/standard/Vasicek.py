"""
Created on Wed Aug  1 15:35:24 2018

@author: jordy


Tests the standard simulated annealing class on the Maximum likelihood
parameter estimation of the Vasicek model.

Gives an example on how to use the accept/reject state and a max step-length
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
from sim_annealing.standard_annealer import Annealer
from MyUtils.printing import title
from MyUtils.simulation import ornstein_uhlenbeck as ous
from MyUtils.estimation import ornstein_uhlenbeck as oue


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

# objective function
actual_solution = np.array([kappa, theta, sigma])
actual_minimum = -oue.loglikelihood_vasicek(actual_solution, data)


def loglikelihood(param, data):
    """
    Annealer can only minimise => maximising ll = minimising -ll
    """
    return -oue.loglikelihood_vasicek(param, data)


# Annealing
x0 = np.ones_like(actual_solution)

# conditions on acceptance of function variables


def accept_reject_state(x):
    """
    Returns boolean, depending on fulfillment of conditions of suggested state
    """
    response = True
    if abs(x[0]) > 10e1:
        response = False
    elif abs(x[1]) > 10e1:
        response = False
    elif (abs(x[2]) > 10e1) or (x[2] < 0):
        response = False
    return response


# It is important to note that whenever real constrictions are put on the
# parameters, that the maximum stepsize (V_max) is adapted accordingly
vasicek = Annealer(x0, loglikelihood, data, None, accept_reject_state)

vasicek.max_updates = 1000
vasicek.V_max = 10e1 * np.ones_like(actual_solution)
own_schedule = vasicek.schedule()
vasicek.set_schedule(own_schedule[1])

results = vasicek.anneal()

title("Vasicek")
results()
results.comparison_known_param(actual_solution, actual_minimum, 5)
