"""
Created on Wed Aug  1 15:35:24 2018

@author: jordy

Scripts contains the Annealer class, making use of the numba package for 
speed advantages 

This Annealer class is able to perform the Simulated Annealing algorithm
as described in:
1. Corana et all. (1987): (CO)
"Minimizing multimodal functions of continuous variables with the simulated
annealing algorithm"
2. Goffe et all. (1994): (GO)
"Global optimization of statistical functions with simmulated annealing"
"""
# =============================================================================
### Packages/Imports
# =============================================================================
import numpy as np
import time
import random
import math

from sim_annealing.jitclass_classes.reportclass import Report
from sim_annealing.jitclass_classes.resultsclass import (
    AnnealerResults,
    AnnealerResults_type,
)

from numba import deferred_type, types, optional
from numba.experimental import jitclass
from collections import OrderedDict


# =============================================================================
### Annealer
# =============================================================================
class Annealer(object):
    """
    This is still a standard class, for the reason that I can print stuff with
    this. See also jitclass below
    """

    def __init__(self, EnergyClass, EC_variables=None, AcceptanceClass=None):

        self.energy = None
        self.energy_type = None

        self.accept = None
        self.accept_type = None

        self.instanciate_EnergyClass(EnergyClass, EC_variables)
        self.instanciate_AcceptanceClass(AcceptanceClass)

        self.report = Report()

    def instanciate_AcceptanceClass(self, AcceptanceClass):
        if AcceptanceClass is not None:
            self.accept = AcceptanceClass()
            self.accept_type = deferred_type()
            self.accept_type.define(AcceptanceClass.class_type.instance_type)
        return None

    def instanciate_EnergyClass(self, EnergyClass, EC_variables):
        if EC_variables is None:
            self.energy = EnergyClass()
        else:
            self.energy = EnergyClass(EC_variables)

        self.energy_type = deferred_type()
        self.energy_type.define(EnergyClass.class_type.instance_type)
        return None

    def _prepare_annealer_dict(self):
        Annealer_spec = OrderedDict()

        Annealer_spec["results"] = AnnealerResults_type
        Annealer_spec["energy"] = self.energy_type
        Annealer_spec["best_energy"] = types.float64
        Annealer_spec["accept_reject"] = optional(self.accept_type)
        Annealer_spec["state"] = types.Array(types.float64, 1, "A")
        Annealer_spec["best_state"] = types.Array(types.float64, 1, "A")
        Annealer_spec["tmax"] = types.float64
        Annealer_spec["tmin"] = types.float64
        Annealer_spec["r_t"] = types.float64
        Annealer_spec["n_eps"] = types.u8
        Annealer_spec["n_T"] = types.u8
        Annealer_spec["n_S"] = types.u8
        Annealer_spec["n"] = types.u8
        Annealer_spec["max_updates"] = types.u8
        Annealer_spec["eps"] = types.float64
        Annealer_spec["V"] = types.Array(types.float64, 1, "A")
        Annealer_spec["c"] = types.Array(types.float64, 1, "A")
        Annealer_spec["V_max"] = types.Array(types.float64, 1, "A")

        return Annealer_spec

    def anneal(
        self,
        x0,
        V_max=None,
        max_updates=None,
        tmax=None,
        tmin=None,
        eps=None,
        n_eps=None,
        n_T=None,
        n_S=None,
        V=None,
        c=None,
        r_t=None,
        auto_schedule=False,
    ):
        start_t = time.time()
        spec = self._prepare_annealer_dict()
        self.x0 = x0

        #######################################################################
        ### Jitclass Annealer
        """
        This is the complicated thing here. In order to use the jitclass/numba
        optionality in a class, you have to specify the types of the 
        members. For this you also need to writ just another 
        jitclass/jitfunction (I opted for the first) to be able to
        use any object function. For this the 'type' of the objectclass 
        given to the annealer should be known, before you write the whole
        Annealer class. Of course this is not the standard order and for this
        reason, I had to write a jitclass, within a standard class. 
        """

        @jitclass(spec)
        class Annealer(object):
            def __init__(self, x0, energy_cl, acc_rej_cl=None):
                self.results = AnnealerResults()
                self.energy = energy_cl
                self.accept_reject = acc_rej_cl
                self.state = self.copy_state(x0)
                self.results.set_initial_value_energy(x0, self.energy.energy(x0))

                # Inititial values
                self.tmax = 250
                self.tmin = 0.001
                self.r_t = 0.85
                self.n_eps = 4
                self.n_S = 20
                self.n = len(x0)
                self.n_T = max(5 * self.n, 100)
                self.max_updates = 10
                self.eps = 10e-5
                self.c = 2 * np.ones_like(x0)
                self.V = np.ones(self.n)
                self.V_max = 10e6 * np.ones(self.n)

            def anneal(self):
                #        self.start_t = time.time()

                # step 0
                T = self.tmax
                k = 0
                current_state = self.copy_state(self.state)
                E = self.energy.energy(self.state)

                self.best_state = self.copy_state(self.state)
                self.best_energy = E

                E_k_star = E * np.ones(self.n_eps + 1)

                while k < self.max_updates and not self.results.convergence:
                    #            print('k: OK')

                    # start N_t loop
                    for m in range(self.n_T):
                        #                print('m: OK')

                        # start N_s loop
                        n_u = np.zeros(self.n)
                        for j in range(self.n_S):
                            #                    print('j: OK')

                            # step 1 + 2
                            for h in range(self.n):
                                #                        print('h: OK')
                                self.move(h)

                                # step 3
                                E_new = self.energy.energy(self.state)
                                self.results.increase_nbr_eval()
                                dE = E_new - E

                                if dE <= 0:
                                    # accept point
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

                        # step 4+5
                        # update V
                        #                print("accept", n_u)
                        #                print("V-pre", self.V)
                        self.adapt_step(n_u)
                    #                print("V_post", self.V)

                    # step 6
                    # adjust T
                    #             print("T", T)
                    T = self.r_t * T

                    E_k_star2 = np.zeros_like(E_k_star)
                    E_k_star2[1:] = E_k_star[:-1].copy()
                    E_k_star2[0] = E

                    E_k_star = E_k_star2.copy()
                    k += 1

                    # step 7
                    if self.check_convergence_criteria(E_k_star):
                        self.results.set_convergence(True)
                    #
                    else:
                        E = self.best_energy
                        current_state = self.copy_state(self.best_state)
                #                        self.state = self.copy_state(current_state)

                # fill results
                self.fill_results()

                return self.results

            def accept_state(self, state):
                if self.accept_reject is None:
                    return True
                else:
                    return self.accept_reject.check(state)

            def adapt_step(self, n_u):
                new_v = np.zeros(self.n)
                for u in range(self.n):
                    if n_u[u] / self.n_S > 0.6:
                        new_v[u] = self.V[u] * (
                            1 + self.c[u] * (n_u[u] / self.n_S - 0.6) / 0.4
                        )
                    elif n_u[u] / self.n_S < 0.4:
                        new_v[u] = self.V[u] / (
                            1 + self.c[u] * (0.4 - n_u[u] / self.n_S) / 0.4
                        )
                    else:
                        new_v[u] = self.V[u]

                for i, v in enumerate(new_v):
                    if v > self.V_max[i]:
                        new_v[i] = self.V_max[i]

                self.V = new_v

                return None

            def copy_state(self, state):
                return np.copy(state)

            def check_convergence_criteria(self, E_k_star):
                last_E = E_k_star[0]
                E_k_star2 = E_k_star[1:].copy()
                for i in range(len(E_k_star2)):
                    E_k_star2[i] = abs(E_k_star2[i] - last_E)

                check_last = abs(last_E - self.best_energy) < self.eps

                if ((E_k_star2 < self.eps).all()) and check_last:
                    return True
                else:
                    return False

            def fill_results(self):
                self.results.set_state_energy(self.best_state, self.best_energy)
                return None

            def move(self, h):
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
                #                    feasible_move = True
                #            print('f', feasible_move, '\n')
                self.state = self.copy_state(suggested_move)
                #        print(suggested_move, self.state)
                return None

            def set_V_max(self, V_max):
                self.V_max = V_max
                return None

            def set_max_updates(self, max_updates):
                self.max_updates = max_updates
                return None

            def schedule(self, steps):
                def run(T, steps):
                    accept, improve = np.zeros(self.n), np.zeros(self.n)
                    E = self.energy.energy(self.state)
                    current_state = self.copy_state(self.state)
                    for _ in range(steps):
                        h = 0
                        while h < self.n:
                            self.move(h)

                            # step 3
                            E_new = self.energy.energy(self.state)
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
                E = self.energy.energy(self.state)
                schedule = np.zeros(2)

                initial_state = self.copy_state(self.state)
                Tmax = np.zeros(self.n)
                Tmin = np.zeros(self.n)

                for h in range(self.n):
                    self.state = initial_state
                    while T[h] == 0.0:
                        self.move(h)
                        T[h] = abs(self.energy.energy(self.state) - E)

                    # Search for Tmax - a temperature that gives 98% acceptance
                    E, acceptance, improvement = run(T[h], steps)

                    while acceptance[h] > 0.99:
                        T[h] = T[h] / 1.5
                        E, acceptance, improvement = run(T[h], steps)

                    #                        print('a-up')

                    while acceptance[h] < 0.95:
                        T[h] = T[h] * 1.5
                        E, acceptance, improvement = run(T[h], steps)
                    #                        print('a-down')

                    Tmax[h] = T[h]

                    # Search for Tmin - a temperature that gives 0% improvement
                    while (improvement[h] > 0.05) and (T[h] > 0.001):
                        T[h] = T[h] / 2.5
                        E, acceptance, improvement = run(T[h], steps)
                    #                        print('i-up')

                    Tmin[h] = T[h]

                    schedule[0] = np.amax(Tmax)
                    schedule[1] = np.amin(Tmin)

                return schedule

            def set_schedule(self, schedule):
                self.tmax = schedule[0]
                self.tmin = schedule[1]
                self.r_t = math.pow(self.tmin / self.tmax, 1 / (0.7 * self.max_updates))
                return None

            ###################################################################

        annealing = Annealer(self.x0, self.energy, self.accept)

        if max_updates is not None:
            annealing.max_updates = max_updates
        if tmax is not None:
            annealing.tmax = tmax
        if tmin is not None:
            annealing.tmin = tmin
        if eps is not None:
            annealing.eps = eps
        if n_eps is not None:
            annealing.n_eps = n_eps
        if V is not None:
            annealing.V = V
        if V_max is not None:
            annealing.V_max = V_max
        if n_T is not None:
            annealing.n_T = n_T
        if n_S is not None:
            annealing.n_S = n_S
        if r_t is not None:
            annealing.r_t = r_t
        if c is not None:
            annealing.c = c
        if auto_schedule:
            schedule = annealing.schedule(2000)
            annealing.set_schedule(schedule)
            print("Scheduling => Done!")

        self.anneal_results = annealing.anneal()
        self.anneal_results.time_elapsed = time.time() - start_t
        self.report.set_results(self.anneal_results)

        return self.anneal_results

    def report_results(self):
        self.report.report_results()
        return None

    def comparison_actual_solution(self, act_param, act_ll, closeness=10e-5):
        self.report.comparison_known_param(act_param, act_ll, closeness)
        return None
