module mainmod
  implicit none

  ! spin-1 model parameters
  real*8,parameter::q_i = 2.2d0
  real*8,parameter::q_f = -2.2d0
  real*8,parameter::c2 = -1.d0
  integer,parameter::Ntot = 1000
  real*8,parameter::dq = 0.1d0

  ! Modified true on-line Watkins's Q-learning
  integer,parameter::dim = int(Ntot/2)+1
  integer max_t_steps
  real*8 total_time
  real*8 delta_time
  complex*16,dimension(dim)::psi_i, psi_f
  integer,parameter::N_tilings = 100
  integer,parameter::N_tiles = 20
  real*8,parameter::q_min = -2.2d0
  real*8,parameter::q_max = 2.2d0
  real*8,dimension(N_tiles)::q_field
  real*8 dq_field
  real*8 state_i
  integer N_realize
  real*8,parameter::alpha_0 = 0.9d0
  real*8,parameter::eta = 0.6d0
  real*8,parameter::lmbda = 1.d0
  real*8,parameter::beta_RL_i = 2.d0
  real*8,parameter::beta_RL_inf = 100.d0
  integer,parameter::T_expl = 20
  real*8,parameter::m_expl = 0.125
  integer,parameter::N_episodes = 20001
  integer,parameter::N_actions = 5
  real*8,dimension(N_actions)::actions

  ! unitary evolution list
  integer,parameter::n1_expm = nint((q_max-q_min)/0.1d0)+1
  integer,parameter::n2_expm = dim
  integer,parameter::n3_expm = dim
  complex*16,dimension(n1_expm, n2_expm, n3_expm)::expm_dict

  ! mission list
  integer,parameter::N_mission = 5
  integer,parameter::N_trials = 20
  integer,dimension(N_mission)::step_ls
  real*8,dimension(N_mission)::time_ls

end module mainmod











