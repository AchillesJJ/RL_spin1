program main
  use ED
  use func
  use Watkins_Q_learning
  use mainmod
  use IFPORT
  implicit none
  real*8 begin, end
  integer i, j, k, cnt

  ! initial randomize
  call random_seed()

  ! initialize quantum state
  psi_i(1) = 1.d0
  psi_f(size(psi_f)) = 1.d0

  ! modified true on-line Watkins's Q-learning parameters
  q_field = linspace(q_min, q_max, N_tiles)
  dq_field = q_field(2)-q_field(1)
  state_i = q_i
  N_realize = 1

  ! preallocate expm_dict
  call unitary_evolve(q_min, q_max, 0.1d0, delta_time, c2, Nt, expm_dict)

  call CPU_TIME(begin)

  !$omp parallel default(shared),private(cnt)
  !$omp do
  do cnt = 1, 8
    ! Q-learning
    call Q_learning(Nt, N_realize, N_episodes, alpha_0, eta, lmbda, beta_RL_i, beta_RL_inf, &
                & T_expl, m_expl, N_tilings, N_tiles, state_i, q_field, dq_field, &
                & max_t_steps, delta_time, q_i, q_f, psi_i, psi_f, expm_dict)
  end do 
  !$omp end do
  !$omp end parallel

  call CPU_TIME(end)

  write(*, *) end-begin

  !**************************!
  !           TEST           !
  !**************************!
  ! real*8,dimension(5)::a, b
  !
  ! a = [1.d0, 2.d0, 3.d0, 4.d0, 5.d0]
  ! write(*, *) a*2.d0

end program main







