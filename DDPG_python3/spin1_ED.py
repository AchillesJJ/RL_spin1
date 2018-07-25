# encoding: utf-8
import time
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint
from scipy.integrate import ode
import math
import cPickle
import cmath

# Hamiltonian of spin-1 system
def Hamiltonian(q, c2, Nt):
    """
    This functions generate the Hamiltonian of spin-1 system
    q: quadratic Zeeman shift
    c2: interaction strength of spin exchange
    Nt: total partical number
    """
    dim = int(Nt/2)+1
    Htot = np.zeros((dim, dim), dtype = np.float64)
    # diagonal part
    for m in range(dim):
        Htot[m, m] = (c2/(2.0*Nt))*(2.0*(Nt-2.0*m)-1)*(2.0*m)-q*(Nt-2.0*m)
    # spin-exchange part
    for m in range(dim-1):
        Htot[m, m+1] = (c2/Nt)*(m+1)*math.sqrt(Nt-2*m-1)*math.sqrt(Nt-2*m)
        Htot[m+1, m] = Htot[m, m+1]

    return Htot
    
# normalization of wave function
def normalize(psi):
    norm = math.sqrt(np.sum(abs(psi)**2))
    return psi/norm

# one-step time evolution matrix
def unitary_op(q, c2, Nt, dt, psi):
    """
    This function generate the evolution matrix with
    q: quadratic Zeeman shift
    c2: interaction strength of spin exchange
    Nt: total partical number
    dt: evolution time interval
    """
    H = Hamiltonian(q[0][0], c2, Nt)
    op = sp.linalg.expm2(-complex(0.0, 1.0)*dt*H)
    psi_f = op.dot(psi)
    return normalize(psi_f)


# if __name__ == '__main__':
#     psi = np.zeros((2,), dtype=np.complex64)
#     psi[0] = 1.0
#     psi2 = unitary_op([[-0.25]], -1.0, 2, 0.5, psi)
#     print(abs(psi2[-1])**2, util.state(psi2))






























