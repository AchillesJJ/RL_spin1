# encoding: utf-8
import numpy as np
import scipy as sp
import cmath
import math
import sys
import os
from spin1_ED import normalize
from random import uniform

# generate random wave function
def random_psi(dim):
    psi = np.array([uniform(-1.0,1.0)+complex(0.0,1.0)*uniform(-1.0,1.0) for i in range(dim)])
    return normalize(psi)

# state-pair
def state(psi):
    """
    This function generates the state pair (rho_0, theta) from wave function psi
    """
    dim = len(psi)
    N = 2*(dim-1)
    nz = np.array([N-2*m for m in range(dim)])
    rho_0 = np.sum(nz*(abs(psi)**2))/N
    ls_L = np.conj(psi[1:dim])
    ls_R = psi[0:-1]
    ele = np.array([math.sqrt((N-2*m)*(N-2*m-1))*(m+1) for m in range(dim-1)])
    res = np.sum((ls_L*ls_R)*ele)
    theta = cmath.phase(res)
    return np.array([rho_0, theta])
    

# feature
def feature(rho_0, theta):
    """
    This function generates feature of state pair (tho_0, theta) using tiles & tilings
    rho_0: spin population on mF = 0 state
    theta: magnetic angle
    """
    return np.array([rho_0, theta])








































