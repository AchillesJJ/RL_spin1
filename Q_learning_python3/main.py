from ED import *
from Watkins_Q_learning import *
import numpy as np
import scipy as sp
import time
import sys
import os
import cPickle
import math

# delta_time = float(sys.argv[1])
# N_realise = int(sys.argv[2])
trial_num = int(sys.argv[1]) 

# spin-1 model parameters
q_i = 3.0 # initial q value
q_f = -3.0 # final q value
c2 = -1.0 # don't change it
Nt = 4
dq = 0.1 # minimum action value of q
delta_time = 0.2 # size of time step

# Modified true on-line Watkins's Q(lambda)-learning

dim = int(Nt/2)+1
max_t_steps = 20
psi_i = np.zeros((dim,), dtype = np.float64)
psi_f = np.zeros((dim,), dtype = np.float64)
psi_i[0] = 1.0 # |0,N,0>
psi_f[-1] = 1.0 # |N/2,0,N/2>
N_tilings = 100
N_tiles = 20
q_min = -3.0 # minimun allowed value of q during protocal
q_max = 3.0 # maximum allowed value of q during protocal
q_field = list(np.linspace(q_min, q_max, N_tiles))
dq_field = q_field[1]-q_field[0]
state_i = np.array([q_i])

N_realise = 1 # realisation  time
alpha_0 = 0.9
eta = 0.6
lmbda = 1.0
beta_RL_i = 2.0
beta_RL_inf = 100.0
T_expl = 20
m_expl = 0.125

# unitary operators list
expm_dict = Unitary_evolve(q_min, q_max, dq, delta_time, c2, Nt)
cPickle.dump(expm_dict, open('input/expm_dict_'+str(Nt)+'_'+str(trial_num)+'.pkl', 'wb'))
print('operator list is done')

average_R_file = open('output/average_R_ep_'+str(trial_num)+'.pkl', 'wb')
all_protocol_file = open('output/all_protocol_'+str(trial_num)+'.pkl', 'wb')
expm_dict = cPickle.load(open('input/expm_dict_'+str(Nt)+'_'+str(trial_num)+'.pkl', 'rb'))

N_episodes = 10001

RL_params = (Nt, N_realise, N_episodes, alpha_0, eta, lmbda, beta_RL_i, beta_RL_inf, T_expl, m_expl, N_tilings, N_tiles, state_i, q_field, dq_field)
phy_params = (max_t_steps, delta_time, q_i, q_f, psi_i, psi_f, expm_dict)
file_path = (all_protocol_file, average_R_file)
Q_learning(*(RL_params+phy_params+file_path), save = False)

average_R_file.close()
all_protocol_file.close()










