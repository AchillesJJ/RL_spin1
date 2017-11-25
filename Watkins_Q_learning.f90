module Watkins_Q_learning
  use func
  implicit none

contains

  !------------------------------------------------------------
  ! This function defines the ramp of RL inverse temperature
  ! input
  ! ts : episode number/time
  ! m  : slope of increase
  ! b  : y intercept
  ! Tl : duration of ramp before zeroing
  !------------------------------------------------------------
  function explore_beta(ts, m, b, Tl, beta_RL_const)
    implicit none
    integer,intent(in)::ts, Tl
    real*8,intent(in)::m, b, beta_RL_const
    real*8 explore_beta

    if(mod((int(ts/Tl)), 2)==1) then
      explore_beta = beta_RL_const
    else
      explore_beta = dble(b+m/2.d0*(dble(1.d0*ts/Tl)-int(ts/Tl)/2.d0))
    end if

  end function explore_beta

  !----------------------------------------------------------
  ! This function finds the feature indices of state S
  !----------------------------------------------------------
  subroutine find_feature_inds(tilings, S, theta_inds)
    implicit none
    real*8,intent(in),dimension(:,:)::tilings
    real*8,intent(in)::S
    integer,intent(inout),dimension(:)::theta_inds
    integer n1, n2, i, j, k, idx
    real*8 left, right

    n1 = size(tilings, 1)
    n2 = size(tilings, 2)
    do i = 1, n1
      call search_sorted(tilings(i, :), S, idx)
      call clip(1, n2-1, idx)
      left = tilings(i, idx)
      right = tilings(i, idx+1)
      if (S-left>right-S) then
        idx = idx+1
      end if
      theta_inds(i) = idx +(i-1)*n2
    end do

  end subroutine find_feature_inds

  !--------------------------------------------------------------
  ! This function biases a given Q function (stored in theta)
  ! to learn a policy by scaling the weights of all actions
  ! down below the value of the best actions
  !--------------------------------------------------------------
  subroutine Learn_Policy(state_i, best_actions, R, theta, tilings, actions)
    implicit none
    real*8,intent(in)::state_i
    real*8,intent(in),dimension(:)::best_actions, actions
    real*8,intent(in)::R
    real*8,intent(inout),dimension(:,:,:)::theta
    real*8,intent(in),dimension(:,:)::tilings
    ! temporary variables
    integer N_tilings, N_tiles, N_actions
    integer t_step, indA, indA_max(1)
    real*8 A, S
    integer,allocatable,dimension(:)::theta_inds, Q

    N_tilings = size(tilings, 1)
    N_tiles = size(tilings, 2)
    N_actions = size(actions)
    allocate(theta_inds(N_tilings))
    allocate(Q(N_actions))
    S = state_i

    do t_step = 1, size(best_actions)
      ! SA index
      A = best_actions(t_step)
      call search(actions, A, indA)
      call find_feature_inds(tilings, S, theta_inds)

      ! check if max learnt
      if (maxval(theta(theta_inds, t_step, :))>dble(R/N_tilings) .and. R>0.d0) then
        Q = sum(theta(theta_inds, t_step, :), 1)
        indA_max = maxloc(Q)

        if (maxval(Q)>1.d-13) then
          theta(theta_inds, t_step, :) = theta(theta_inds, t_step, :)*R/Q(indA_max(1))
        end if
      end if

      ! calculate theta function
      theta(theta_inds, t_step, indA) = dble((R+1.d-2)/N_tilings)
      S = S+A
    end do

    deallocate(theta_inds)
    deallocate(Q)

  end subroutine Learn_Policy

  !-----------------------------------------------------
  ! This function builds a protocol from actions set
  !-----------------------------------------------------
  subroutine build_protocol(actions_taken, q_i, delta_t, protocol, t_ls)
    implicit none
    real*8,intent(in),dimension(:)::actions_taken
    real*8,intent(in)::q_i, delta_t
    real*8,intent(inout),dimension(:)::protocol, t_ls
    integer i, j, k
    real*8 A, S

    S = q_i
    do i = 1, size(actions_taken)
      A = actions_taken(i)
      S = S+A
      protocol(i) = S
      t_ls(i) = (i-1)*delta_t
    end do

  end subroutine build_protocol

  !--------------------------------------------------------
  ! This function gives the reward (fidelity on TF state)
  !--------------------------------------------------------
  function fidelity(psi)
    implicit none
    complex*16,intent(in),dimension(:)::psi
    real*8 fidelity

    fidelity = abs(psi(size(psi)))**2

  end function fidelity

  !---------------------------------------------------------
  ! Modified true on-line Watkins's Q-learning algorithm
  !---------------------------------------------------------
  subroutine Q_learning(Ntot, N_realize, N_episodes, alpha_0, eta, lmbda, beta_RL_i, beta_RL_inf, &
                      & T_expl, m_expl, N_tilings, N_tiles, state_i, q_field, dq_field, &
                      & max_t_steps, delta_time, q_i, q_f, psi_i, psi_f, expm_dict)
    implicit none
    ! input & output parameters
    integer,intent(in)::Ntot, N_realize, N_episodes, T_expl, N_tilings, N_tiles, max_t_steps
    real*8,intent(in)::alpha_0, eta, beta_RL_i, beta_RL_inf, m_expl, state_i, delta_time
    real*8,intent(in)::q_i, q_f, lmbda, dq_field
    real*8,intent(in),dimension(:)::q_field
    complex*16,intent(in),dimension(:)::psi_i, psi_f
    complex*16,intent(inout),dimension(:,:,:)::expm_dict
    ! temporary variables
    real*8,allocatable,dimension(:,:,:)::theta
    real*8,allocatable,dimension(:,:)::tilings
    real*8,allocatable,dimension(:)::fire_trace, u0, u, Q, actions_taken, avail_actions, &
                                   & best_actions, alpha, protocol, t_ls
    complex*16,allocatable,dimension(:)::psi
    integer,allocatable,dimension(:)::theta_inds, avail_inds, theta_inds_prime
    real*8,allocatable,dimension(:,:,:)::theta_old, e
    real*8,dimension(5)::actions
    integer,dimension(5)::avail_inds_tmp
    integer N_actions, cnt, ep, t_step, avail_inds_cnt, indA
    real*8 q_shift, best_R, R, S, S_prime, beta_RL, A_greedy, A, delta_t, &
         & Q_old, delta_q

    ! preallocate physical state and define actions
    allocate(psi(size(psi_i)))
    psi = cmplx(0.d0, 0.d0)
    actions = [-12.d0, -6.d0, 0.d0, 6.d0, 12.d0]
    N_actions = size(actions)

    ! preallocate theta and tilings
    allocate(theta(N_tiles*N_tilings, max_t_steps, N_actions))
    allocate(tilings(N_tilings, N_tiles))

    ! preallocate weight vector theta and tilings
    theta = 0.d0
    allocate(theta_old(size(theta,1),size(theta,2),size(theta,3)))
    theta_old = theta
    do cnt = 1, N_tilings
      call random_number(q_shift)
      tilings(cnt, :) = q_field+q_shift*dq_field
    end do

    ! preallocate eligibility trace
    allocate(e(size(theta,1),size(theta,2),size(theta,3)))
    e = 0.d0
    allocate(fire_trace(N_tilings))
    fire_trace = 1.d0

    ! preallocate usage vector : inverse gradient descent learning rate
    allocate(u0(N_tiles*N_tilings))
    allocate(u(N_tiles*N_tilings))
    u = 0.d0
    u0 = 1.d0/alpha_0

    ! preallocate reward and feature index
    best_R = -1.d0
    R = 0.d0
    allocate(theta_inds(N_tilings))
    allocate(theta_inds_prime(N_tilings))
    theta_inds = 0

    ! preallocate Q function
    allocate(Q(N_actions))
    Q = 0.d0

    ! preallocate action taken list
    allocate(actions_taken(max_t_steps))

    ! preallocate best action
    allocate(best_actions(max_t_steps))

    ! preallocate alpha
    allocate(alpha(N_tilings))

    !-------------------------------
    !  Loop over multiple episodes
    !-------------------------------
    do ep = 1, N_episodes

      e = 0.d0 ! init eligibility trace to zero
      u = u0 ! init learning rate
      S = state_i ! init state S at t = 0

      ! get feature of S
      call find_feature_inds(tilings, S, theta_inds)

      ! init Q(S,t=0,:) for all actions
      Q = sum(theta(theta_inds, 1, :), 1)

      ! preallocate physical state
      psi = psi_i

      ! init taken actions
      actions_taken = 0.d0

      ! define learning temperature
      beta_RL = explore_beta(ep-1, m_expl, beta_RL_i, T_expl, beta_RL_inf)

      !---------------------------------------
      !   Loop over time step in episide ep
      !---------------------------------------
      do t_step = 1, max_t_steps

        ! available actions for state S at time t = t_step
        avail_inds_cnt = 0
        do cnt = 1, N_actions
          if (S+actions(cnt)<=q_field(size(q_field)) .and. S+actions(cnt)>=q_field(1)) then
            avail_inds_cnt = avail_inds_cnt+1
            avail_inds_tmp(avail_inds_cnt) = cnt
          end if
        end do
        allocate(avail_inds(avail_inds_cnt))
        allocate(avail_actions(avail_inds_cnt))
        avail_inds(:) = avail_inds_tmp(1:avail_inds_cnt)
        avail_actions = actions(avail_inds)

        ! determine greedy action for S at time t = t_step
        if (beta_RL<beta_RL_inf) then
          if (mod(ep-1, 2)==0) then
            A_greedy = avail_actions(random_choice_from_max(Q(avail_inds)))
          else
            A_greedy = best_actions(t_step)
          end if
        else
          A_greedy = avail_actions(random_choice_from_max(Q(avail_inds)))
        end if

        ! choose action for S at time t = t_step by explore & exploit phase
        if (beta_RL<beta_RL_inf) then ! explore by Boltzman-policy
          A = avail_actions(random_choice_from_BP(exp(beta_RL*Q(avail_inds))))

          ! reset eligibility trace to 0 if A is exploratory
          if (abs(A-A_greedy)>1.d-12) then
            e = 0.d0
          end if
        else ! exploit
          A = A_greedy
        end if

        ! index of A in actions
        call search(actions, A, indA)

        ! record action taken
        actions_taken(t_step) = A

        !------------------------------------
        ! determine reward by time evolution
        !------------------------------------
        S_prime = S
        S_prime = S_prime+A ! next state S' at t = t_step+1
        ! physical state at t = t_step+1
        psi = matmul(expm_dict(nint((S_prime-minval(q_field))/0.1d0)+1, :, :), psi)
        call normalize(psi)

        ! asign reward
        R = 0.d0 ! non-terminal state
        if (t_step==max_t_steps) then ! terminal state
          R = R+fidelity(psi)
        end if

        ! learning rate update
        u(theta_inds) = (1.d0-eta)*u(theta_inds)
        u(theta_inds) = u(theta_inds)+1.d0
        alpha = 1.d0/(N_tilings*u(theta_inds))

        !--------------------------------------
        ! Q-learning TD error and update rule
        !--------------------------------------
        delta_t = R-Q(indA) ! R(t+1)-Q(St,A)
        Q_old = sum(theta(theta_inds, t_step, indA)) ! Q(S,t,A)
        e(theta_inds, t_step, indA) = alpha*fire_trace ! e(S,t,A)<-1

        ! check if S_prime is terminal or out of range
        if (t_step==max_t_steps) then
          ! update theta
          theta = theta+delta_t*e
          ! GD error in field q
          delta_q = Q_old-sum(theta(theta_inds, t_step, indA))
          theta(theta_inds, t_step, indA) = theta(theta_inds, t_step, indA) &
                                          & +delta_q*alpha
          ! deallocate temporary variables inside loop
          deallocate(avail_actions)
          deallocate(avail_inds)
          ! go to next episode
          exit
        end if

        ! feature of S_prime
        call find_feature_inds(tilings, S_prime, theta_inds_prime)

        ! t-dependent Watkins's Q-learning
        Q = sum(theta(theta_inds_prime, t_step+1, :), 1) ! Q(S',t+1,:)

        ! update theta
        delta_t = delta_t+maxval(Q) ! max_a[Q(S',t+1,a)] for all a
        theta = theta+delta_t*e

        ! TD error in field q
        delta_q = Q_old-sum(theta(theta_inds, t_step, indA))
        theta(theta_inds, t_step, indA) = theta(theta_inds, t_step, indA) &
                                        & +delta_q*alpha

        ! update trace e(S,t,A) and e(:,:,:)
        e(theta_inds, t_step, indA) = e(theta_inds, t_step, indA) &
                                    & -sum(e(theta_inds, t_step, indA))*alpha
        e = lmbda*e

        ! update state and its feature
        S = S_prime
        theta_inds = theta_inds_prime

        ! deallocate temporary variables inside loop
        deallocate(avail_actions)
        deallocate(avail_inds)

      end do

      ! best policy so far
      if (R-best_R>1.d-12) then
        ! write(*, *) "best fidelity is : ", R, "with episodes : ", ep
        ! update list best action so far
        best_actions = actions_taken
        ! update best reward and weight vector theta
        best_R = R
        call Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)
      end if

      ! force-learn replay every 100 episides
      if ((mod(ep+1, 2*T_expl)-T_expl==0) .and. (ep .ne. 1) .and. (ep .ne. N_episodes)) then
        call Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)
      elseif ((mod(int(ep/T_expl), 2)==1) .and. (abs(R-best_R)>1.d-12)) then
        call Learn_Policy(state_i, best_actions, best_R, theta, tilings, actions)
      end if

      ! check convergence of theta function
      ! write(*, '(I4,3F12.6)') ep, beta_RL, R, maxval(abs(theta-theta_old))
      theta_old = theta

    end do

    allocate(protocol(max_t_steps))
    allocate(t_ls(max_t_steps))
    call build_protocol(actions_taken, state_i, delta_time, protocol, t_ls)
    ! write(*, *) protocol

    ! deallocate temporary variables outside loop
    deallocate(psi)
    deallocate(theta)
    deallocate(tilings)
    deallocate(theta_old)
    deallocate(e)
    deallocate(fire_trace)
    deallocate(u0)
    deallocate(u)
    deallocate(theta_inds)
    deallocate(theta_inds_prime)
    deallocate(Q)
    deallocate(actions_taken)
    deallocate(best_actions)
    deallocate(alpha)
    deallocate(protocol)
    deallocate(t_ls)

  end subroutine Q_learning


end module Watkins_Q_learning
















































