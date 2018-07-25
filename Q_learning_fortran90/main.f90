program main
  use ED
  use func
  use Watkins_Q_learning
  use mainmod
  implicit none
  real*8 begin, end
  integer i, j, k

  ! initial randomize, this is very important
  call random_seed()

  ! initialize quantum state
  psi_i(1) = 1.d0
  psi_f(size(psi_f)) = 1.d0

  ! modified true on-line Watkins's Q-learning parameters
  q_field = linspace(q_min, q_max, N_tiles)
  dq_field = q_field(2)-q_field(1)
  state_i = q_i
  N_realize = 1

  ! actions list
  ! actions = [-12.d0, -6.d0, 0.d0, 6.d0, 12.d0]
  ! actions = [-6.d0,-3.d0,-2.d0,-1.d0,-0.4d0,-0.2d0,-0.1d0,0.d0,0.1d0,0.2d0,0.4d0,1.d0,2.d0,3.d0,6.d0]
  actions = [-4.4d0, -2.2d0, 0.d0, 2.2d0, 4.4d0]
  
  ! preallocate step_ls and time_ls
  time_ls = [10.d0, 12.d0, 15.d0, 20.d0, 25.d0]
  step_ls = [100, 100, 100, 100, 100]
  
  ! preallocate output file path
  open(100, file = 'output/best_reward.dat', status = 'replace')
  open(200, file = 'output/best_protocol.dat', status = 'replace')
  close(100)
  close(200)
  
  ! call CPU_TIME(begin)
  do i = 1, N_mission
    
    ! preallocate max_t_steps, total_time and delta_time
    total_time = dble(time_ls(i))
    max_t_steps = int(step_ls(i))
    delta_time = dble(total_time/max_t_steps)
    
    ! preallocate unitary evolution operator list 
    call unitary_evolve(q_min, q_max, 0.1d0, delta_time, c2, Ntot, expm_dict)
    
    ! each mission tries N_trials time 
    do j = 1, N_trials 
      ! Q-learning
      call Q_learning()
    end do 
    
    write(*, *) "mission", i, "accomplished"
    
  end do 
  ! call CPU_TIME(end)
  ! write(*, *) end-begin

end program main































