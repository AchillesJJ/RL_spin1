from ED import *
import numpy as np
import numpy.random as random
import time
import os
import sys
import pickle
import cPickle

def explore_beta(t, m, b, T, beta_RL_const = 1000.0):
	"""
	This function defines the ramp of the RL inverse temperature:

	t: episode number/time
	m: slope of increase
	b: y intercept
	T: duration of ramp before zeroing
	"""
	if (t//T)%2==1:
		return beta_RL_const
	else:
		return b + m/2.0*(float(t)/T - (t//T)/2.0)


def find_feature_inds(tilings, S, theta_inds):
	"""
	This function finds the feature indices of s state S in the set of tilings.
	"""
	for _k, tiling in enumerate(tilings): # cython-ise this loop as inline fn!
		idx = tiling.searchsorted(S)
		idx = np.clip(idx, 1, len(tiling)-1)
		left, right = tiling[idx-1], tiling[idx]
		idx -= S - left < right - S
		theta_inds[_k]=idx[0]+_k*tilings.shape[1]
	return theta_inds


def Learn_Policy(state_i, best_actions, R, theta, tilings, actions):
	"""
	This function biases a given Q function (stored in the linear approximant theta)
	to learn a policy by scaling the weights of all actions down below the value of the
	best action.
	"""
	N_tilings = tilings.shape[0]
	N_tiles = tilings.shape[1]
	# preallocate theta_inds
	theta_inds_zeros = np.zeros((N_tilings,), dtype=int)
	# init state at t=0
	S = state_i.copy()

	for t_step, A in enumerate(best_actions):
		# SA index
		indA = np.searchsorted(actions, A)
		theta_inds = find_feature_inds(tilings, S, theta_inds_zeros)

		# check if max learnt
		if max(theta[theta_inds,t_step,:].ravel())>R/N_tilings and R>0: # weights
			Q = np.sum(theta[theta_inds,t_step,:],axis=0)
			indA_max=np.argmax(Q)

			if max(Q) > 1E-13:
				theta[theta_inds,t_step,:]*=R/Q[indA_max]

		# calculate theta function
		theta[theta_inds,t_step,indA] = (R+1E-2)/(N_tilings)

		S+=A

	#print 'force-learned best encountered policy'
	return theta


def build_protocol(actions, q_i, delta_t):
	""" This function builds a protocol from the set of actions """
	protocol = np.zeros_like(actions)
	t = np.array([delta_t*_i for _i in range(len(actions))])
	S = q_i
	for _i,A in enumerate(actions):
		S += A
		protocol[_i] = S
	return protocol, t


def greedy_protocol(theta, tilings, actions, q_i, delta_t, max_t_steps, q_field):
	""" This function builds the best encounteres protocol from best_actions """
	protocol = np.zeros((max_t_steps,), dtype=np.float64)
	t = np.array([delta_t*_i for _i in range(max_t_steps)])
	S = np.array([q_i])

	N_tilings = tilings.shape[0]
	N_tiles = tilings.shape[1]

	# preallocate theta_inds
	theta_inds_zeros = np.zeros((N_tilings,), dtype=int)

	for t_step in range(max_t_steps):

		avail_inds = np.argwhere((S[0]+np.array(actions)<=q_field[-1])*(S[0]+np.array(actions)>=q_field[0])).squeeze()
		avail_actions = actions[avail_inds]

		# calculate Q(s,a)
		theta_inds = find_feature_inds(tilings,S,theta_inds_zeros)
		Q = np.sum(theta[theta_inds,t_step,:],axis=0)
		# find greedy action
		A_greedy = avail_actions[np.argmax(Q[avail_inds])]

		S[0] += A_greedy
		protocol[t_step] = S

	return protocol, t


def reward(psi):
	""" This function gives the reward according to spin population """
	Ntot = int((len(psi)-1)*2)
	dim = int(Ntot/2)+1
	ls = np.array([i*1.0 for i in range(dim)], dtype=np.float64)
	rw = 2.0*np.sum(ls*(abs(psi)**2))/Ntot

	return rw

def fidelity(psi):
	""" This function gives the reward according to fidelity on twin-Fock state  """
	rw = abs(psi[-1])**2

	return rw

def weight_fid(psi):
	"""
	This function gives the reward according to the wighted fidelity on the final
	ten percents of basis satate around the twin-Fock state
	"""
	dim = len(psi)
	gamma = 0.9 # discount rate
	num = 100 # the last 100 basis states are considered
	rw = 0.0
	for cnt, i in enumerate(range(dim-num, dim)):
		rw += (gamma**(num-1-cnt))*(abs(psi[i])**2)
	rw = (1.0-gamma)*rw

	return rw

#----------------------------------------------------------
#	Modified true on-line Watkins's Q(lambda) algorithm
#----------------------------------------------------------
def Q_learning(Ntot, N,N_episodes,alpha_0,eta,lmbda,beta_RL_i,beta_RL_inf,T_expl,m_expl,N_tilings,N_tiles,state_i,q_field,dq_field,
			   max_t_steps,delta_time,q_i,q_f,psi_i,psi_f,expm_dict, file1,file2,
			   theta=None,tilings=None,save=False):
	"""
	This function applies modified Watkins' Q-Learning for time-dependent states with
	force-learn replays.

	1st row: RL arguments
	2nd row: physics arguments
	3rd row: optional arguments
	"""

	# global variable

	# preallocate unitary operator list
	# expm_dict = cPickle.load(open('input/expm_dict_'+str(N)+'_'+str(Ntot)+'.pkl', 'rb'))

	# preallocate physical state
	psi = np.zeros_like(psi_i)
	# define actions
	pos_actions = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0]
	neg_actions = [-i for i in pos_actions]
	actions = np.sort(neg_actions+[0.0]+pos_actions)
	N_actions = len(actions)

	# preallocate unitary operator list

	# preallocate weight vector theta
	if theta is None:
		theta=np.zeros((N_tiles*N_tilings, max_t_steps, N_actions), dtype=np.float64)
	theta_old=theta.copy()
	if tilings is None:
		tilings = np.array([q_field+np.random.uniform(0.0,dq_field,1) for j in xrange(N_tilings)])

	# preallocate eligibility trace
	e = np.zeros_like(theta) # for all state, time, actions
	fire_trace = np.ones(N_tilings) # for update SAP

	# pre-allocate usage vector: inverse gradient descent learning rate
	u0 = 1.0/alpha_0*np.ones((N_tiles*N_tilings,), dtype=np.float64)
	u=np.zeros_like(u0)

	# preallocate quantities
	Return_ave = np.zeros((N_episodes,),dtype=np.float64)
	Return = np.zeros_like(Return_ave)
	Fidelity_ep = np.zeros_like(Return_ave)
	protocol_ep = np.zeros((Fidelity_ep.shape[0],max_t_steps),)

	best_R = -1.0 # init best reward
	R = 0.0 # init reward
	theta_inds_zeros = np.zeros((N_tilings,), dtype=int)

	#-----------------------------------
	#	Loop over multiple episodes
	#-----------------------------------
	for ep in xrange(N_episodes):

		e *= 0.0 # init eligibility trace to zero
		u[:] = u0[:] # init learning rate
		S = state_i.copy() # init state S at t=0

		# get feature of S
		theta_inds = find_feature_inds(tilings, S, theta_inds_zeros)

		# init Q(S,t=0,:) for all actions
		Q = np.sum(theta[theta_inds, 0, :], axis = 0)

		# preallocate physical state
		psi[:] = psi_i[:]

		# store taken actions
		actions_taken = np.zeros((max_t_steps,), dtype=np.float64)

		# define learning temperature
		beta_RL = explore_beta(ep,m_expl,beta_RL_i,T_expl,beta_RL_const=beta_RL_inf)

		# explore status
		explored = False

		#---------------------------------------
		#	Loop over time step in episode ep
		#---------------------------------------
		for t_step in xrange(max_t_steps):

			# avaliable actions for state S at time t=t_step
			avail_inds = np.argwhere((S[0]+np.array(actions)<=q_field[-1])*(S[0]+np.array(actions)>=q_field[0])).squeeze()
			avail_actions = actions[avail_inds]

			# determine greedy action for S at time t=t_step
			if beta_RL<beta_RL_inf:
				if ep%2==0:
					A_greedy = avail_actions[random.choice(np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]
				else:
					A_greedy = best_actions[t_step]
			else:
				A_greedy = avail_actions[random.choice(np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).ravel() ) ]

			# choose action for S at time t=t_step by explore & exploit phase
			if beta_RL<beta_RL_inf: # explore by Boltzman-policy
				P = np.exp(beta_RL*Q[avail_inds])
				A = avail_actions[np.searchsorted(np.cumsum(P/np.sum(P)),random.uniform(0.0,1.0))]

				# reset eligibility trace to 0 if A is exploratory
				if abs(A-A_greedy)>np.finfo(A).eps:
					e *= 0.0
			else: # exploit
				A = A_greedy

			# index of A in actions
			indA = np.searchsorted(actions, A)

			# record action taken
			actions_taken[t_step] = A

			#-----------------------------------------
			#	determine the reward by evolution
			#-----------------------------------------
			S_prime = S.copy()
			S_prime[0] += A # next state S' at t=t_step+1
			# physical state at t=t_step+1
			b = S_prime[0]
			#psi = exp_H.dot(psi)
			psi=expm_dict[int(np.rint((b-min(q_field))/min(pos_actions)))].dot(psi)

			# asign reward
			R *= 0.0 # non-terminal state
			if t_step==max_t_steps-1: # terminal state
				# R += fidelity
				# R += abs(psi.conj().dot(psi_f))**2
				R += fidelity(psi)
				# R += fidelity(psi)
				# R += weight_fid(psi)

			#----------------------------------------
			#			learning rate update
			#----------------------------------------
			u[theta_inds] *= (1.0-eta)
			u[theta_inds] += 1.0
			alpha = 1.0/(N_tilings*u[theta_inds]) # update learning rate

			#-----------------------------------
			# Q(lambda) TD error and update rule
			#-----------------------------------
			delta_t = R-Q[indA] # R(t+1)-Q(St,A)
			Q_old = theta[theta_inds, t_step, indA].sum() # Q(S,t,A)
			e[theta_inds, t_step, indA] = alpha*fire_trace # e(S,t,A)<-1

			# check if S_prime is terminal or out of range
			if t_step==max_t_steps-1:
				# update theta
				theta += delta_t*e
				# GD error in field q
				delta_q = Q_old - theta[theta_inds, t_step, indA].sum()
				theta[theta_inds, t_step, indA] += alpha*delta_q
				# go to next episode
				break

			# feature of S_prime
			theta_inds_prime=find_feature_inds(tilings, S_prime, theta_inds_zeros)

			# t-dependent Watkins's Q-learning
			Q = np.sum(theta[theta_inds_prime,t_step+1,:],axis=0) # Q(S',t+1,:)

			# update theta
			delta_t += np.max(Q) # max_a(Q(S',t+1,a)) for all a
			theta += delta_t*e

			# TD error in field q
			delta_q = Q_old - theta[theta_inds, t_step, indA].sum()
			theta[theta_inds, t_step, indA] += alpha*delta_q

			# update traces e(S,t,A) and e(:,:,:)
			e[theta_inds, t_step, indA] -= alpha*e[theta_inds, t_step, indA].sum()
			e *= lmbda

			# update state and its feature
			S[:] = S_prime[:]
			theta_inds[:] = theta_inds_prime[:]

		# best policy so far
		if R-best_R>1E-12:
			print("best encountered fidelity is {} with training number {}".format(np.around(R, 4), ep))
			# update list of best actions so far
			best_actions = actions_taken[:]
			# update best reward and weight vector theta
			best_R = R
			theta = Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)

		# force-learn replay every 100 episodes
		if ((ep+1)%(2*T_expl)-T_expl==0 and ep not in [0,N_episodes-1]): # and beta_RL<20.0:
			theta = Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)
		elif (ep//T_expl)%2==1 and abs(R-best_R)>1E-12:
			theta = Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)

		# # check convergence of Q-function
		# print ep, "beta_RL,R,d_theta:", beta_RL, R, np.max(abs(theta.ravel()-theta_old.ravel()))
		# theta_old=theta.copy()

		# # record quantities
		# Return_ave[ep] = 1.0/(ep+1)*(R+ep*Return_ave[ep-1])
		# Return[ep] = R
		# Fidelity_ep[ep] = R
		# protocol_ep[ep,:] = build_protocol(actions_taken, state_i[0], delta_time)[0].astype(int)

		# if (ep+1)%(2*T_expl)==0:
		# 	print "finished simulating episode {} with fidelity {} at hx_f = {}.".format(ep+1,np.round(R,5),S_prime[0])
		# 	#print 'best encountered fidelity is {}.'.format(np.round(best_R,5))
		# 	#print 'current inverse exploration tampeature is {}.'.format(np.round(beta_RL,3))

		# if (ep%100==0): # ouput every 400 episodes
		# 	# print('episodes {} with reward {}'.format(ep, best_R))
		# 	cPickle.dump(best_R, file2)

	# # determine the global best protocol and fidelity
	# protocol_best, t_best = build_protocol(best_actions, state_i[0], delta_time)
	# protocol_greedy, t_greedy = greedy_protocol(theta, tilings, actions, state_i[0], delta_time, max_t_steps, q_field)
    # 
	# out_data = {}
	# out_data[0] = max_t_steps*delta_time
	# out_data[1] = best_R
	# out_data[2] = protocol_best
	# cPickle.dump(out_data, file1)
	# # cPickle.dump(best_R, file2)
	# print('best reward is {} with {} episodes'.format(best_R, N_episodes-1))








