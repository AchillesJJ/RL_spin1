module mainmod
  implicit none

  ! spin-1 model parameters
  real*8,parameter::q_i = 6.d0
  real*8,parameter::q_f = -6.d0
  real*8,parameter::c2 = -1.d0
  integer,parameter::Nt = 2
  real*8,parameter::dq = 0.1d0

  ! Modified true on-line Watkins's Q-learning
  integer,parameter::dim = int(Nt/2)+1
  integer,parameter::max_t_steps = 50
  real*8,parameter::total_time = 2.2d0
  real*8,parameter::delta_time = total_time/max_t_steps
  complex*16,dimension(dim)::psi_i, psi_f
  integer,parameter::N_tilings = 100
  integer,parameter::N_tiles = 20
  real*8,parameter::q_min = -6.d0
  real*8,parameter::q_max = 6.d0
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
  integer,parameter::N_episodes = 101

  ! unitary evolution list
  integer,parameter::n1_expm = nint((q_max-q_min)/0.1d0)+1
  integer,parameter::n2_expm = dim
  integer,parameter::n3_expm = dim
  complex*16,dimension(n1_expm, n2_expm, n3_expm)::expm_dict

  ! ! theta and tilings
  ! real*8,dimension(N_tiles*N_tilings, max_t_steps, 5)::theta
  ! real*8,dimension(N_tiles, N_tilings)::tilings

end module mainmod











