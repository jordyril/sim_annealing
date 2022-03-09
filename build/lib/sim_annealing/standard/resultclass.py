# =============================================================================
# Packages
# =============================================================================
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import random
import time
import scipy.optimize as sco
import scipy.stats as scs
import collections as col
from support_functions.printing import time_string
# =============================================================================
# Standard Annealer results class
# =============================================================================
class AnnealerResult(object):
    def __init__(self):
        self.func = None
        self.termination = None
        self.state = None
        self.energy = None
        self.time_elapsed = None
        self.nbr_eval = None
        self.x0 = None
        self.f0 = None
        self.other_var = None
        self.annealing_param = None
        self.annealing_succes = None
    
    def __call__(self):
        dash = '-' * 40
        if self.termination == True:
            print("{:<40s}".format("Algorithm Converged"))
        else:
            print('{:<40s}'.format("Max iterations reached"))
        print(dash)
        print('{:<25s}'.format('x0:'), self.x0)
        print('{:<25s}'.format('f0:'), self.f0)
        print('{:<25s}'.format('Optimal State:'), self.state)
        print('{:<25s}'.format('Optimal Energy:'), self.energy)
        print('{:<25s}'.format('# func evaluations:'), self.nbr_eval)
        
                   
        print('{:<25s}{:<23s}'.format('Total time elapsed:', 
              time_string(self.time_elapsed)), '\n')
    
        return None
    
        
    def comparison_known_param(self, act_param, act_ll, closeness=None):
        dash = '-' * 40
        if closeness is None: limit = self.annealing_param['eps']
        else: limit = closeness
        
        self.distance_optimal_f = act_ll - self.energy
        self.distance_optimal_p =  act_param - self.state
        self.annealing_succes = False
        if ((abs(self.distance_optimal_f) <= limit) and 
            (np.all(abs(self.distance_optimal_p) <= limit))):
            self.annealing_succes = True
            
        print('Comparison with actual solution')
        print(dash)
        if self.annealing_succes:
            print('SUCCESSFUL annealing - (limit =', str(limit) + ')')
        else:
            print("UNSUCCESSFUL annealing - (limit =", str(limit) + ')')
        print('{:<55s}'.format("Distance from (real) optimal function value:"), 
              self.distance_optimal_f)
        if abs(self.distance_optimal_f) > limit:
            print(55 * ' ', '=> Something is WRONG\n')
        
        print('{:<55s}'.format("Distance from (real) optimal" +
              " function parameters:"), 
              self.distance_optimal_p)
        if not np.all(abs(self.distance_optimal_p) <= limit):
            print(55 * ' ', '=> Something is WRONG')
    
        return None
