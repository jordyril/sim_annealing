"""
Created on Tue Jul 31 12:52:05 2018

@author: jordy
"""
# =============================================================================
# Packages
# =============================================================================
from numba import jitclass, types, deferred_type, optional
from collections import OrderedDict

# =============================================================================
# ANNEALERRESULTS CLASS
# =============================================================================
## ANNEALERRESULTS class start
AnnealerResults_spec = OrderedDict()
AnnealerResults_type = deferred_type()

AnnealerResults_spec['convergence'] =  types.boolean
AnnealerResults_spec['state'] =  optional(types.Array(types.float64, 1, 'A'))
AnnealerResults_spec['energy'] =  optional(types.float64)
AnnealerResults_spec['initial_value'] =  optional(types.Array(types.float64, 1, 'A'))
AnnealerResults_spec['time_elapsed'] =  optional(types.float64)
AnnealerResults_spec['nbr_eval'] =  types.u8
AnnealerResults_spec['initial_energy'] =  optional(types.float64)

@jitclass(AnnealerResults_spec)
class AnnealerResults(object):
    """Class that serves for summarizing results form simulated annealing"""
    def __init__(self):
        self.convergence = False
        self.state = None
        self.energy = None
        self.initial_value = None
        self.initial_energy = None
        self.time_elapsed = None
        self.nbr_eval = 1
    
    def increase_nbr_eval(self):
        self.nbr_eval += 1
        return None
            
    def get_convergence(self):
        return self.convergence
    
    def get_initial_value_energy(self):
        return self.initial_value, self.initial_energy
        
    def get_state_energy(self):
        return self.state, self.energy
    
    def get_time_elapsed(self):
        return self.time_elapsed
   
    def set_convergence(self, convergence):
        self.convergence = convergence
        return None
    
    def set_initial_value_energy(self, x0, f0):
        self.initial_value = x0
        self.initial_energy = f0
        return None
        
    def set_state_energy(self, state, energy):
        self.state = state
        self.energy = energy 
        return None
    
    def set_time_elapsed(self, seconds):
        self.time_elapsed = seconds
        return None 
