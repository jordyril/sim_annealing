"""
Created on Tue Jul 31 13:15:49 2018

@author: jordy

Script contains the Annealer class in its standard, most basic form.
This Annealer class is able to perform the Simulated Annealing algorithm
as described in:
1. Corana et all. (1987): (CO)
"Minimizing multimodal functions of continuous variables with the simulated
annealing algorithm"
2. Goffe et all. (1994): (GO) APPENDIX B
"Global optimization of statistical functions with simulated annealing"
"""
# =============================================================================
# Packages
# =============================================================================
import numpy as np
import math
import random
import time

from sim_annealing._utils import time_string
from sim_annealing.standard_classes.resultclass import AnnealerResult

# =============================================================================
# Standard Annealer class
# =============================================================================


class Annealer(object):
    def __init__(self, x_0, func, other_var=None, step_sizes=None, accept_state=None):
        self.results = AnnealerResult()  # Resultclass to store specific results
        self.state = self.copy_state(x_0)  # curent values of parameters
        self.results.func = func
        self.results.other_var = other_var  # extra parameters needed
        self.V = (
            step_sizes if step_sizes is not None else np.ones_like(self.state)
        )  # standard initial stepsize
        self.check_admissability_state = accept_state
        self.nbr_eval = 0
        self.start = time.time()

        # initial values
        self.Tmax = 250  # starting themperature
        self.Tmin = 0.001  # end themperature
        self.r_t = 0.85  # speed of decrease in T
        self.N_eps = 4  # speed of algorithm convergence
        self.N_S = 20  # nbr of steps made with same stepsize
        self.n = len(self.state)  # nbr of variables
        self.N_T = max(5 * self.n, 100)  # nbr of step adaptations at same T
        self.max_updates = 10  # nbr of T drops
        self.eps = 10e-5  # speeed of algorithm converges
        self.terminate = False  # convergence
        self.c = 2 * np.ones_like(self.state)
        self.V_max = 10e22 * np.ones(self.n)  # max stepsize

        self.results.x0 = self.copy_state(x_0)
        self.results.f0 = self.energy()

    def set_schedule(self, schedule):
        """
        Sets the schedule and adapts r_t to this new schedule
        """
        self.Tmax = schedule["tmax"]
        self.Tmin = schedule["tmin"]
        self.max_updates = schedule["updates"]
        self.r_t = min(
            0.85, math.pow(self.Tmin / self.Tmax, 1 / (0.5 * self.max_updates))
        )
        return None

    def set_Ns(self, schedule):
        self.N_eps = schedule["n_eps"]
        self.N_S = schedule["n_s"]
        self.N_T = schedule["n_t"]
        return None

    def set_eps(self, eps):
        self.eps = eps
        return None

    def accept_state(self, a_state):
        """
        Check if suggested solution ensures (optional) feasibility
        """
        if self.check_admissability_state is None:
            return True
        else:
            return self.check_admissability_state(a_state)

    def set_V_max(self, V_max):
        self.V_max = V_max
        return None

    def check_convergence_criteria(self, E_k_star):
        """
        Checks the criteria og algo convergence
        """
        last_E = E_k_star[0]
        E_k_star2 = E_k_star[1:].copy()
        for i in range(len(E_k_star2)):
            E_k_star2[i] = abs(E_k_star2[i] - last_E)

        check_last = abs(last_E - self.best_energy) < self.eps

        if ((E_k_star2 < self.eps).all()) and check_last:
            return True
        else:
            return False

    def adapt_step(self, n_u):
        """
        Adapting the stepsize for each parameter
        """
        new_v = np.zeros(self.n)
        for u in range(0, self.n):
            if n_u[u] / self.N_S > 0.6:
                new_v[u] = self.V[u] * (1 + self.c[u] * (n_u[u] / self.N_S - 0.6) / 0.4)
            elif n_u[u] / self.N_S < 0.4:
                new_v[u] = self.V[u] / (1 + self.c[u] * (0.4 - n_u[u] / self.N_S) / 0.4)
            else:
                new_v[u] = self.V[u]

        for i, v in enumerate(new_v):
            if v > self.V_max[i]:
                new_v[i] = self.V_max[i]

        self.V = new_v

        return None

    def energy(self):
        """
        Evaluates object function at current state
        """
        if self.results.other_var is None:
            return self.results.func(self.state)
        else:
            return self.results.func(self.state, self.results.other_var)

    def move(self, h):
        """
        Makes a move in the value of the h'th parameter
        """
        feasible_move = False
        while feasible_move is False:
            #            print('f', feasible_move)
            r = np.random.uniform(-1, 1)
            #            print('r', r)
            suggested_move = self.copy_state(self.state)
            #            print(suggested_move, self.state)
            suggested_move[h] = suggested_move[h] + r * self.V[h]
            #            print(suggested_move, self.state)
            feasible_move = self.accept_state(suggested_move)
        #            print('f', feasible_move, '\n')
        self.state = self.copy_state(suggested_move)
        #        print(suggested_move, self.state)
        return None

    def copy_state(self, state):
        """
        hardcopies a state
        """
        return state.copy()

    def progress_update(self, T, result_list, time_s, k):
        """
        Intermediate report, handy for debugging
        """
        time_dif_total = time.time() - self.start
        time_dif = time.time() - time_s
        dash = "-" * 50
        print("Intermediate report", k, "out of", self.max_updates)
        print(
            "Time elapsed:",
            time_string(time_dif),
            "(Total:",
            time_string(time_dif_total),
            ")",
        )
        print(dash)
        print("{:<25s}".format("Current T:"), T)
        print("{:<25s}".format("Current V:"), self.V)
        print("{:<25s}".format("Current list:"), result_list)
        print("{:<25s}".format("Current state:"), self.state)
        print("{:<25s}".format("Current optimal energy:"), self.best_energy)
        print("{:<25s}".format("Current optimal state:"), self.best_state, "\n")
        return None

    def fill_results_object(self):
        """
        Preparing the final results
        """
        self.results.termination = self.terminate
        self.results.state = self.best_state
        self.results.energy = self.best_energy
        self.results.nbr_eval = self.nbr_eval
        self.results.time_elapsed = time.time() - self.start
        self.results.annealing_param = {
            "Tmax": self.Tmax,
            "Tmin": self.Tmin,
            "V": self.V,
            "V_max": self.V_max,
            "N_S": self.N_S,
            "N_T": self.N_T,
            "N_eps": self.N_eps,
            "eps": self.eps,
        }
        return None

    def anneal(self, visual_progress=True):
        """
        Performs simulated ammealing algorithm. Implementation of the algorithm
        as stated in Corana, A. et al. (1987).
        """
        self.start = time.time()

        # step 0
        T = self.Tmax
        k = 0
        current_state = self.copy_state(self.state)
        E = self.energy()
        self.nbr_eval += 1

        self.best_state = self.copy_state(self.state)
        self.best_energy = E

        E_k_star = E * np.ones(self.N_eps + 1)

        # start annealing
        while k < self.max_updates and not self.terminate:

            time_s = time.time()
            m = 0
            # start N_t loop - constant T
            while m < self.N_T:

                # start N_s loop - constant V
                n_u = np.zeros(self.n)  # stores acceptance/improvement counts
                j = 0
                while j < self.N_S:

                    # step 1 + 2
                    h = 0
                    while h < self.n:

                        self.move(h)

                        # step 3
                        E_new = self.energy()

                        self.nbr_eval += 1
                        dE = E_new - E

                        if dE <= 0:
                            # Accept point
                            E = E_new
                            current_state = self.copy_state(self.state)
                            n_u[h] += 1
                            if (E < self.best_energy) and (
                                self.accept_state(self.state)
                            ):
                                self.best_energy = E
                                self.best_state = self.copy_state(self.state)
                        else:
                            # Metropolis criterium
                            p = math.exp(-dE / T)
                            p_random = random.random()
                            if p > p_random:
                                # accept point
                                E = E_new
                                current_state = self.copy_state(self.state)
                                n_u[h] += 1
                            else:
                                self.state = self.copy_state(current_state)

                        # step 4
                        h += 1
                    j += 1

                # step 5
                # update V
                self.adapt_step(n_u)

                m += 1

            # step 6
            # adjust T

            T = self.r_t * T

            E_k_star2 = np.zeros_like(E_k_star)
            E_k_star2[1:] = E_k_star[:-1].copy()
            E_k_star2[0] = E

            E_k_star = E_k_star2.copy()
            k += 1

            # step 7
            if self.check_convergence_criteria(E_k_star):
                self.terminate = True

            else:
                E = self.best_energy
                current_state = self.copy_state(self.best_state)

            if visual_progress:
                self.progress_update(T, E_k_star, time_s, k)

        # filling results object
        self.fill_results_object()

        return self.results

    def schedule(self, steps=2000):
        """
        Looks for 'optimal' annealing schedule
        """

        def constant_T_run(T, steps):
            """
            Annealing over constant T and fixed amount of steps
            """
            accept, improve = np.zeros(self.n), np.zeros(self.n)
            E = self.energy()
            current_state = self.copy_state(self.state)
            for _ in range(steps):
                h = 0
                while h < self.n:
                    self.move(h)

                    # step 3
                    E_new = self.energy()
                    dE = E_new - E

                    if dE <= 0:
                        # accept point
                        E = E_new
                        current_state = self.copy_state(self.state)
                        accept[h] += 1
                        improve[h] += 1
                    else:
                        # Metropolis criterium
                        p = math.exp(-dE / T)
                        p_random = random.random()
                        if p > p_random:
                            # accept point
                            E = E_new
                            current_state = self.copy_state(self.state)
                            accept[h] += 1
                        else:
                            self.state = self.copy_state(current_state)

                    h += 1

            return E, accept / steps, improve / steps

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = np.zeros(self.n)
        E = self.energy()
        schedules = {}

        initial_state = self.copy_state(self.state)
        Tmax = np.zeros(self.n)
        Tmin = np.zeros(self.n)
        for h in range(self.n):
            step = 0
            self.state = initial_state
            while T[h] == 0.0:
                step += 1
                self.move(h)
                T[h] = abs(self.energy() - E)

            # Search for Tmax - a temperature that gives 98% acceptance
            E, acceptance, improvement = constant_T_run(T[h], steps)

            step += steps
            while acceptance[h] > 0.99:
                T[h] = T[h] / 1.5
                E, acceptance, improvement = constant_T_run(T[h], steps)
                step += steps

            while acceptance[h] < 0.95:
                T[h] = T[h] * 1.5
                E, acceptance, improvement = constant_T_run(T[h], steps)
                step += steps

            Tmax[h] = T[h]

            # Search for Tmin - a temperature that gives only 5% improvement
            while (improvement[h] > 0.05) and (T[h] > 0.001):
                T[h] = T[h] / 2.5
                E, acceptance, improvement = constant_T_run(T[h], steps)
                step += steps

            Tmin[h] = T[h]

            schedules[h] = {
                "tmax": Tmax[h],
                "tmin": Tmin[h],
                "updates": self.max_updates,
            }

        # Don't perform anneal, just return params
        schedule = {
            "tmax": schedules[0]["tmax"],
            "tmin": schedules[0]["tmin"],
            "updates": self.max_updates,
        }

        for h in range(1, self.n):
            if schedules[h]["tmax"] > schedule["tmax"]:
                schedule["tmax"] = schedules[h]["tmax"]
            if schedules[h]["tmin"] < schedule["tmin"]:
                schedule["tmin"] = schedules[h]["tmin"]

        return schedules, schedule
