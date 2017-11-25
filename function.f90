module func
  implicit none

contains

  !---------------------------------
  ! search in sorted list with clip
  !---------------------------------
  subroutine search_sorted(tiling, S, idx)
    implicit none
    real*8,intent(in),dimension(:)::tiling
    real*8,intent(in)::S
    integer,intent(inout)::idx
    integer n1, i, j, k

    n1 = size(tiling)
    if (tiling(1)>=S) then
      idx = 0
    elseif (tiling(n1)<S) then
      idx = n1
    else
      do i = 1, n1-1
        if (tiling(i)<S .and. tiling(i+1)>=S) then
          idx = i
          exit
        end if
      end do
    end if

  end subroutine search_sorted

  !------------------------------
  ! clip to given range (ni, nf)
  !------------------------------
  subroutine clip(ni, nf, idx)
    implicit none
    integer,intent(in)::ni, nf
    integer,intent(inout)::idx

    if (idx<ni) then
      idx = ni
    elseif (idx>nf) then
      idx = nf
    end if

  end subroutine clip

  !---------------------------------------
  ! search a given element in sorted list
  !---------------------------------------
  subroutine search(ls, S, idx)
    implicit none
    real*8,intent(in),dimension(:)::ls
    real*8,intent(in)::S
    integer,intent(inout)::idx
    integer i

    do i = 1, size(ls)
      if (ls(i)==S) then
        idx = i
        exit
      end if
    end do

  end subroutine


  !--------------------------------------------------------
  ! given idx randomly through the max value sublist of ls
  !--------------------------------------------------------
  function random_choice_from_max(ls)
    implicit none
    real*8,intent(in),dimension(:)::ls
    integer random_choice_from_max
    integer,dimension(size(ls))::ls_max_ind
    real*8 max_val, rand_p
    integer i, j, k, cnt

    max_val = maxval(ls)
    cnt = 0
    do i = 1, size(ls)
      if (ls(i)==max_val) then
        cnt = cnt+1
        ls_max_ind(cnt) = i
      end if
    end do
    ! determine random choice probability
    call random_number(rand_p)
    ! output
    random_choice_from_max = ls_max_ind(int(rand_p*cnt)+1)

  end function

  !----------------------------------------------
  ! randomly choose from Q-value Boltzman-policy
  !----------------------------------------------
  function random_choice_from_BP(ls)
    implicit none
    real*8,intent(in),dimension(:)::ls
    integer random_choice_from_BP
    real*8,dimension(size(ls))::ls_norm
    real*8,dimension(size(ls)+1)::cumsum_ls
    integer i, j, k, cnt
    real*8 cumsum, rand_p

    ls_norm = ls/sum(ls)
    ! preallocate cumlative sum list
    cumsum_ls(1) = 0.d0
    cumsum = 0.d0
    do i = 1, size(ls)
      cumsum = cumsum+ls_norm(i)
      cumsum_ls(i+1) = cumsum
    end do

    ! random choice from cumsum_ls
    call random_number(rand_p)
    call search_sorted(cumsum_ls, rand_p, random_choice_from_BP)

  end function random_choice_from_BP

  !--------------------------------------
  ! normalize wave function at each step
  !--------------------------------------
  subroutine normalize(ls)
    implicit none
    complex*16,intent(inout),dimension(:)::ls

    ls = ls/sum(abs(ls)**2)

  end subroutine normalize

  !------------------------------------
  ! mimick python's linspace function
  !------------------------------------
  function linspace(val_i, val_f, num)
    implicit none
    real*8,intent(in)::val_i, val_f
    integer,intent(in)::num
    real*8,dimension(num)::linspace
    real*8 dv
    integer i, j, k

    dv = dble((val_f-val_i)/(num-1))
    do i = 1, num
      linspace(i) = val_i+(i-1)*dv
    end do

  end function

end module func





















