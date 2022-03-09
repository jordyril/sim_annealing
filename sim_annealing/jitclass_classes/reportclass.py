# =============================================================================
# Packages
# =============================================================================
import numpy as np

from MyUtils.printing import time_string


# =============================================================================
### Report CLASS
# =============================================================================


class Report(object):
    def __init__(self, an_results=None):
        self.results = an_results

    def comparison_known_param(self, act_param, act_ll, closeness=10e-6):
        dash = "-" * 40
        limit = closeness

        distance_optimal_f = self.results.energy - act_ll
        distance_optimal_p = self.results.state - act_param
        self.annealing_succes = False
        if (abs(distance_optimal_f) <= limit) and (
            np.all(abs(distance_optimal_p) <= limit)
        ):
            self.annealing_succes = True

        print("Comparison with actual solution")
        print(dash)
        if self.annealing_succes:
            print("SUCCESSFUL annealing - (limit =", str(limit) + ")")
        else:
            print("UNSUCCESSFUL annealing - (limit =", str(limit) + ")")
        print(
            "{:<55s}".format("Distance from (real) optimal function value:"),
            distance_optimal_f,
        )
        if abs(distance_optimal_f) > limit:
            print(55 * " ", "=> Something is WRONG\n")

        print(
            "{:<55s}".format("Distance from (real) optimal" + " function parameters:"),
            distance_optimal_p,
        )
        if not np.all(abs(distance_optimal_p) <= limit):
            print(55 * " ", "=> Something is WRONG")

        print("\n")
        return None

    def report_results(self):
        dash = "-" * 40
        if self.results.convergence:
            print("{:<40s}".format("Algorithm Converged"))
        else:
            print("{:<40s}".format("Max iterations reached"))
        print(dash)
        print("{:<25s}".format("x0:"), self.results.initial_value)
        print("{:<25s}".format("f0:"), self.results.initial_energy)
        print("{:<25s}".format("Optimal State:"), self.results.state)
        print("{:<25s}".format("Optimal Energy:"), self.results.energy)
        print("{:<25s}".format("# func evaluations:"), self.results.nbr_eval)
        if self.results.time_elapsed is not None:
            time_str = time_string(self.results.time_elapsed)
            print("{:<25s}".format("Total time elapsed:"), time_str)
        print("\n")
        return None

    def set_results(self, an_results):
        self.results = an_results
        return None
