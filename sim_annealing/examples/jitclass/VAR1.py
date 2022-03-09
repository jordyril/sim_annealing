"""
Created on Thu Aug  2 11:49:29 2018

@author: jordy
This script shows the example of the the Maximum likelihood parameter 
estimation of a VAR(1) process. Also, the example illustrates the use of njit
functions together with jitclass.
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


from sim_annealing.jitclass_annealer import Annealer
from MyUtils.printing import title
from MyUtils.simulation import ornstein_uhlenbeck as ous
from MyUtils.estimation import normal
from MyUtils.estimation.general import check_within_ci


from MyUtils.vector import vec, vech, duplication_matrix, sqrtMatrix
from collections import OrderedDict
from numba import types, jitclass, njit

import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================
# Has to be written in njit style for future use in this script
@njit
def devec(elements, shape):
    """
    Reverses the vec operation. In case no shape is given, a square matrix gets
    returned based on the lenght of th elements-array
    """
    rows = shape[0]
    cols = shape[1]
    A = np.zeros((rows, cols))

    e = 0
    for i in range(cols):
        for j in range(rows):
            A[j][i] = elements[e]
            e += 1

    return A


# =============================================================================
###  2 Factor - VAR(1)
# X_{t+1} = nu + A1 @ X_t + Eps_{t+1}, with Eps_{t+1} ~ N(0, Omega)
# X = BZ + U
# Follows p 69-93 from Lutkepohl (2007)
# "New introduction to multiple time series analysis"
# All equations marked with a reference, refer to these pages
# =============================================================================
# =============================================================================
# DATA SIMULATION
# =============================================================================
times = 150
k = 2
p = 1  # var(p) process

# simulate dataset - var1 process data
nu1 = 0.0
nu2 = 0.0
nu = np.array([[nu1], [nu2]])

a111 = 0.076278548
a112 = 0.0
a121 = -0.189995715
a122 = 0.352497604
A1 = np.array([[a111, a112], [a121, a122]])

sigma11 = 1.0
sigma22 = 1.0
corr12 = -0.0
Omega = np.array(
    [
        [sigma11 ** 2, corr12 * sigma11 * sigma22],
        [corr12 * sigma11 * sigma22, sigma22 ** 2],
    ]
)

eps = np.random.multivariate_normal(np.zeros(2), Omega, times).T

t = np.zeros_like(eps)
t0 = np.array([[0.0], [0.0]])
t[:, 0] = t0.T

for i in range(1, times):
    t[:, i] = (nu + A1 @ t[:, i - 1].reshape((2, 1)) + eps[:, i].reshape((2, 1))).T

# plot
plt.plot(t[0], label="t1")
plt.plot(t[1], label="t2")
plt.show()

# =============================================================================
### Parameter estimation
# =============================================================================
mu = la.inv(np.identity(k) - A1) @ nu  # actual mu, unknown in reality, # 3.3.13
Y0 = np.subtract(t[:, p:], mu)  # 3.3.2

mu_hat = t[:, p:].mean(axis=1).reshape((k, 1))  # estimated proces mean # 3.3.9
Y0_hat = np.subtract(t[:, p:], mu_hat)  # hat => computed with mu_hat # 3.3.2

A = A1  # actual A  # 3.3.2
alpha = vec(A)  # actual alpha (vectorization of A) # 3.3.2

X = np.subtract(t[:, :-p], mu)  # 3.3.2
X_hat = np.subtract(t[:, :-p], mu_hat)  # hat: computed with mu_hat # 3.3.2

y0_hat = vec(Y0_hat)  # 3.3.2

### MLE - actual maximization
title("MLE")

# .trace() does not work in njit environment, so have to write function myself
@njit
def trace(x):
    n = x.shape[0]
    trc = 0.0
    for i in range(n):
        trc += x[i, i]
    return trc


@njit
def loglikelihood_gaussian_VARp(param, t):
    """
    Computes the loglikelihood function of a k-factor Var(p) process given 
    the parameters and the data.
    Based on 3.4.5 in Lutkepohl(2007)
    parameters consists of an array with the first k elements being the 
    estimates for mu (k process averages), remaining elements of the 
    parameters array are the estimates for kxpk A matrix.
    """
    k = t.shape[0]
    p = int((len(param) - k) / k ** 2)

    mu = devec(param[:k], (k, 1))

    A_is = param[k:]  # elements for A matrix

    A = devec(A_is, (k, k * p))

    Y = t[:, p:]  # all 'unknown' observations

    T = Y.shape[1]

    Y0 = np.subtract(Y, mu)

    X = np.subtract(t[:, :-p], mu)

    U = Y0 - A @ X

    sigma_U = U @ U.T / T

    # Note the pseudo inverse - after weeks of struggle, I found out that the
    # normal inv and numba does not do well for 'big' numbers, giving
    # completely wrong results after annealing. This is (hopefully) solved
    # by using the pseudo-inv
    ll = (
        -1 / 2 * k * T * np.log(2 * np.pi)
        - T / 2 * np.log(la.det(sigma_U))
        - trace(U.T @ la.pinv(sigma_U) @ U)
    )

    return ll, sigma_U


# Annealer
var1_spec = OrderedDict()
var1_spec["data"] = types.Array(types.float64, 2, "A")


@jitclass(var1_spec)
class GaussianVAR1(object):
    def __init__(self, data):
        self.data = data

    def energy(self, param):
        return -loglikelihood_gaussian_VARp(param, self.data)[0]


@jitclass([])
class VAR1Accept(object):
    def __init__(self):
        pass

    def check(self, state):
        return True


# actual values
actual_solution = np.hstack((mu.flatten(), A1.flatten(order="F")))
actual_minimum = GaussianVAR1(t).energy(actual_solution)

x0 = np.ones_like(actual_solution)  # initial guess

var1 = Annealer(GaussianVAR1, t, VAR1Accept)
results = var1.anneal(x0, max_updates=500)

title("VAR(1)")
var1.report_results()
var1.comparison_actual_solution(actual_solution, actual_minimum, 5)

# estimates
mu_mll = devec(var1.anneal_results.state[:k], (k, 1))
A1_mll = devec(var1.anneal_results.state[k:], (k, p * k))
nu_mll = (np.identity(k) - A1_mll) @ mu_mll

