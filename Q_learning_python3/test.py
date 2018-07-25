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
from ED import *
from tqdm import tqdm

qls = np.linspace(-5.0, 0.0, 100)

# for _i, q in enumerate(qls):
#     H1 = Hamiltonian(q, 1.0, 100)
#     w1, v1 = sp.linalg.eigh(H1)
#     gs1 = v1[:,0]
#     H2 = Hamiltonian(q, -1.0, 100)
#     w2, v2 = sp.linalg.eigh(H2)
#     gs2 = v2[:,0]
#     print('q = {}, overlap is {}'.format(q, abs(gs1.dot(np.conj(gs2)))**2))


H2 = Hamiltonian(0.1, 1.0, 100)
w2, v2 = sp.linalg.eigh(H2)
gs2 = abs(v2[:,0])**2
print(gs2)
