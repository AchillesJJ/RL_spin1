#-----------------------------------------------------------------
# This module generate spin-1 Hamiltonian and evolution operator
# for multiple time steps which are stored in output file
#-----------------------------------------------------------------

import time
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint
from scipy.integrate import ode
from math import *
import cPickle
import cmath

# Hamiltonian of spin-1 system
def Hamiltonian(q, c2, Nt):
    """
    This functions generate the Hamiltonian of spin-1 system
    q: quadratic Zeeman shift
    c2 : interaction strength of spin exchange
    Nt : total partical number
    """
    dim = int(Nt/2)+1
    Htot = np.zeros((dim, dim), dtype = np.float64)
    # diagonal part
    for m in range(dim):
        Htot[m, m] = (c2/(2.0*Nt))*(2.0*(Nt-2.0*m)-1)*(2.0*m)-q*(Nt-2.0*m)
    # spin-exchange part
    for m in range(dim-1):
        Htot[m, m+1] = (c2/Nt)*(m+1)*sqrt(Nt-2*m-1)*sqrt(Nt-2*m)
        Htot[m+1, m] = Htot[m, m+1]

    return Htot

# generate unitary operator list
def Unitary_evolve(q_min, q_max, dq, dt, c2, Nt):
    """
    This functions generate the unitary operator list
    q_min : minimum value of q
    q_max : maximum value of q
    dq : minimum defference step value of q
    dt : const evolution time step value
    c2 : interaction strength of spin exchange
    Nt : total partical number
    """
    expm_dict = {} # dict of unitary operators
    dim = int((q_max-q_min)/dq)
    cnt = int(Nt/2)+1
    for i in range(dim + 1):
        qval = q_min + i * dq
        Htot = Hamiltonian(qval, c2, Nt)
        # expm_dict[i] = sp.linalg.expm2(-complex(0,1)*dt*Htot)
        w, v = sp.linalg.eigh(Htot)
        diag = np.zeros((cnt, cnt), dtype=np.complex64)
        for j, val in enumerate(w):
            diag[j, j] = cmath.exp(-complex(0, 1) * dt * val)
        expm_dict[i] = (v).dot(diag).dot(v.transpose())
        print('mission qavl = {} is done'.format(qval))

    return expm_dict


# ode routine
def fcn(t, y, qval, c2, Nt):
    dim = int(Nt/2)+1
    # real part (0-dim-1) and imaginary part(dim-2*dim-1)
    ydot = np.zeros((2*dim,), dtype=np.float64)
    for i in range(dim):
        ydot[i] = ((c2/(2.0*Nt))*(2.0*(Nt-2.0*i)-1)*(2.0*i)-q*(Nt-2.0*i))*y[i+dim]
        ydot[i+dim] = -((c2/(2.0*Nt))*(2.0*(Nt-2.0*i)-1)*(2.0*i)-q*(Nt-2.0*i))*y[i]
        if i != dim-1:
            ydot[i] += ((c2/Nt)*(i+1)*sqrt(Nt-2*i-1)*sqrt(Nt-2*i))*y[i+1+dim]
            ydot[i+dim] += -((c2/Nt)*(i+1)*sqrt(Nt-2*i-1)*sqrt(Nt-2*i))*y[i+1]
        if i != 0:
            ydot[i] += ((c2/Nt)*(i)*sqrt(Nt-2*(i-1)-1)*sqrt(Nt-2*(i-1)))*y[i-1+dim]
            ydot[i+dim] += -((c2/Nt)*(i)*sqrt(Nt-2*(i-1)-1)*sqrt(Nt-2*(i-1)))*y[i-1]

    return ydot

def time_evolve(dt, q, c2, Nt, psi_i):
    psi_f = np.zeros_like(psi_i)
    dim = int(Nt/2)+1
    init = np.array([np.real(psi_i), np.imag(psi_i)]).ravel()
    # sol = odeint(fcn, init, [0.0, dt], args=(q, c2, Nt), full_output=1, mxstep=5000000)
    sol = ode(fcn)
    sol.set_integrator('vode',nsteps=500000,method='bdf')
    sol.set_initial_value(init,0.0)
    sol.set_f_params(q, c2, Nt)
    sol.integrate(sol.t+dt)
    psi_f = sol.y[0:dim]+complex(0.0, 1.0)*sol.y[dim:2*dim]

    return psi_f

#--------------------------------
#          SIMPLE TEST
#--------------------------------
Htot = Hamiltonian(-0.5, -1.0, 1000)
w, v = sp.linalg.eigh(Htot)
gs = v[:,0]
rho = np.zeros((3,3), dtype=np.float64)
rho[0, 0] = np.sum(np.asarray([]) * abs(v[:,0])**2)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 