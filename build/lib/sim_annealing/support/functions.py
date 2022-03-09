"""
Created on Tue Jul 31 13:27:59 2018

@author: jordy
"""

# =============================================================================
# Functions 
# =============================================================================
def time_string(seconds):
    """Returns time in seconds as a string formatted HH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%2i:%02i:%02i' % (h, m, s)
