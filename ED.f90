!-----------------------------------------------
! This module generate spin-1 Hamiltonian and
! list of unitary evolution matrix
!-----------------------------------------------

module ED
  use func
  implicit none

contains

  subroutine build_H(q, c2, Nt, Htot)
    implicit none
    real*8,intent(in)::q, c2
    integer,intent(in)::Nt
    real*8,intent(inout),dimension(:,:)::Htot
    integer i, j, k, dim

    ! initialize Htot to zero
    Htot = 0.d0
    dim = int(Nt/2)+1
    do i = 1, dim-1
      Htot(i, i) = (c2/2.d0/Nt)*(2.d0*(Nt-2.d0*i)+3)*2.d0*(i-1)-q*(Nt-2.d0*i+2.d0)
      Htot(i, i+1) = (c2/Nt)*i*sqrt(Nt-2.d0*i+1.d0)*sqrt(Nt-2.d0*i+2.d0)
      Htot(i+1, i) = Htot(i, i+1)
    end do
    Htot(dim, dim) = (c2/2.d0/Nt)*(2.d0*(Nt-2.d0*dim)+3)*2.d0*(dim-1)-q*(Nt-2.d0*dim+2.d0)
    return

  end subroutine build_H

  subroutine unitary_evolve(q_min, q_max, dq, dt, c2, Nt, expmls)
    implicit none
    real*8,intent(in)::q_min, q_max, dq, dt, c2
    integer,intent(in)::Nt
    complex*16,intent(inout),dimension(:,:,:)::expmls
    ! Eigen-solver parameters
    character*1::jobz = 'V'
    character*1::uplo = 'U'
    integer::nv
    integer::lda
    integer::lwork
    real*8,allocatable,dimension(:)::work
    integer::info
    ! temporary variables
    real*8,allocatable,dimension(:,:)::uop, Htot
    real*8,allocatable,dimension(:)::egv
    complex*16,allocatable,dimension(:,:)::diag
    integer::dim, cnt, i, j, k
    real*8::qval

    ! build Hamiltonian
    dim = int(Nt/2)+1
    cnt = nint((q_max-q_min)/dq)+1
    nv = dim
    lda = nv
    lwork = 3*nv-1
    allocate(work(lwork))
    allocate(diag(dim, dim))
    allocate(Htot(dim, dim))
    allocate(uop(dim, dim))
    allocate(egv(dim))
    do i = 1, cnt
      qval = q_min+(i-1)*dq
      ! build Hamiltonian
      call build_H(qval, c2, Nt, Htot)
      uop = Htot
      ! matrix diagonlization
      call dsyev(jobz, uplo, nv, uop, lda, egv, work, lwork, info)
      diag = 0.d0
      do j = 1, dim
        diag(j, j) = exp(-cmplx(0.d0, 1.d0)*dt*egv(j))
      end do
      ! calculate matrix exponential
      expmls(i, :, :) = matmul(matmul(uop, diag), transpose(uop))
    end do

    deallocate(work)
    deallocate(diag)
    deallocate(Htot)
    deallocate(uop)
    deallocate(egv)

  end subroutine unitary_evolve

end module ED




















