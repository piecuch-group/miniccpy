module dipeom4_p
    
      use omp_lib

      implicit none

      contains
              subroutine build_hr1(resid,&
                                   r4_amps,r4_excits,&
                                   h2_oovv,&
                                   n4,& 
                                   no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: h2_oovv(no,no,nu,nu)
                  integer, intent(in) :: r4_excits(n4,6)
                  real(kind=8), intent(in) :: r4_amps(n4)

                  real(kind=8), intent(inout) :: resid(no,no)
                  !f2py intent(in,out) :: resid(0:no-1,0:no-1)

                  real(kind=8) :: val, rval
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet

                  do idet = 1,n4
                     rval = r4_amps(idet)
                     ! x1(ij) <- A(ij/mn) v(mnef)*r3(efijmn)
                     e = r4_excits(idet,1); f = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); m = r4_excits(idet,5); n = r4_excits(idet,6);
                     resid(i,j) = resid(i,j) + h2_oovv(m,n,e,f)*rval ! (1)
                     resid(j,m) = resid(j,m) + h2_oovv(i,n,e,f)*rval ! (im)
                     resid(j,n) = resid(j,n) + h2_oovv(m,i,e,f)*rval ! (in)
                     resid(i,m) = resid(i,m) - h2_oovv(j,n,e,f)*rval ! (jm)
                     resid(i,n) = resid(i,n) - h2_oovv(m,j,e,f)*rval ! (jn)
                     resid(m,n) = resid(m,n) + h2_oovv(i,j,e,f)*rval ! (im)(jn)
                  end do

                  ! antisymmetrize A(ij)
                  do i = 1,no
                     do j = i+1,no
                        resid(i,j) = resid(i,j) - resid(j,i)
                        resid(j,i) = -resid(i,j)
                     end do
                  end do
                  do i = 1,no
                     resid(i,i) = 0.0d0
                  end do

              end subroutine build_hr1

              subroutine build_hr2(resid,&
                                   r4_amps,r4_excits,&
                                   h1_ov,h2_vovv,h2_ooov,&
                                   n4,& 
                                   no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: h1_ov(no,nu)
                  real(kind=8), intent(in) :: h2_vovv(nu,no,nu,nu)
                  real(kind=8), intent(in) :: h2_ooov(no,no,no,nu)
                  integer, intent(in) :: r4_excits(n4,6)
                  real(kind=8), intent(in) :: r4_amps(n4)

                  real(kind=8), intent(inout) :: resid(no,no,nu,no)
                  !f2py intent(in,out) :: resid(0:no-1,0:no-1,0:nu-1,0:no-1)

                  real(kind=8) :: val, rval
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet

                  do idet = 1,n4
                     rval = r4_amps(idet)
                     ! x2(ijck) <- A(m/ijk)A(ce) h1(me)*r3(ceijkm)
                     c = r4_excits(idet,1); e = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); m = r4_excits(idet,6);
                     ! (1)
                     resid(i,j,c,k) = resid(i,j,c,k) + h1_ov(m,e)*rval ! (1)
                     resid(j,k,c,m) = resid(j,k,c,m) - h1_ov(i,e)*rval ! (im)
                     resid(i,k,c,m) = resid(i,k,c,m) + h1_ov(j,e)*rval ! (jm)
                     resid(i,j,c,m) = resid(i,j,c,m) - h1_ov(k,e)*rval ! (km)
                     ! (ce)
                     resid(i,j,e,k) = resid(i,j,e,k) - h1_ov(m,c)*rval ! (1)
                     resid(j,k,e,m) = resid(j,k,e,m) + h1_ov(i,c)*rval ! (im)
                     resid(i,k,e,m) = resid(i,k,e,m) - h1_ov(j,c)*rval ! (jm)
                     resid(i,j,e,m) = resid(i,j,e,m) + h1_ov(k,c)*rval ! (km)
                     ! x2(ijck) <- A(ij/mn)A(cf) -h2(mnkf)*r3(cfijmn)
                     c = r4_excits(idet,1); f = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); m = r4_excits(idet,5); n = r4_excits(idet,6);
                     ! (1)
                     resid(i,j,c,:) = resid(i,j,c,:) - h2_ooov(m,n,:,f)*rval ! (1)
                     resid(j,m,c,:) = resid(j,m,c,:) - h2_ooov(i,n,:,f)*rval ! (im)
                     resid(j,n,c,:) = resid(j,n,c,:) - h2_ooov(m,i,:,f)*rval ! (in)
                     resid(i,m,c,:) = resid(i,m,c,:) + h2_ooov(j,n,:,f)*rval ! (jm)
                     resid(i,n,c,:) = resid(i,n,c,:) + h2_ooov(m,j,:,f)*rval ! (jn)
                     resid(m,n,c,:) = resid(m,n,c,:) - h2_ooov(i,j,:,f)*rval ! (im)(jn)
                     ! (cf)
                     resid(i,j,f,:) = resid(i,j,f,:) + h2_ooov(m,n,:,c)*rval ! (1)
                     resid(j,m,f,:) = resid(j,m,f,:) + h2_ooov(i,n,:,c)*rval ! (im)
                     resid(j,n,f,:) = resid(j,n,f,:) + h2_ooov(m,i,:,c)*rval ! (in)
                     resid(i,m,f,:) = resid(i,m,f,:) - h2_ooov(j,n,:,c)*rval ! (jm)
                     resid(i,n,f,:) = resid(i,n,f,:) - h2_ooov(m,j,:,c)*rval ! (jn)
                     resid(m,n,f,:) = resid(m,n,f,:) + h2_ooov(i,j,:,c)*rval ! (im)(jn)
                     ! x2(ijck) <- A(n/ijk) h2(cnef)*r3(efijkn)
                     e = r4_excits(idet,1); f = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); n = r4_excits(idet,6);
                     resid(i,j,:,k) = resid(i,j,:,k) + h2_vovv(:,n,e,f)*rval ! (1)
                     resid(j,k,:,n) = resid(j,k,:,n) - h2_vovv(:,i,e,f)*rval ! (in)
                     resid(i,k,:,n) = resid(i,k,:,n) + h2_vovv(:,j,e,f)*rval ! (jn)
                     resid(i,j,:,n) = resid(i,j,:,n) - h2_vovv(:,k,e,f)*rval ! (kn)
                  end do

                  ! antisymmetrize A(ijk)
                  do i = 1,no-2
                     do j = i+1,no-1
                        do k = j+1,no
                           resid(i,j,:,k) = resid(i,j,:,k) - resid(i,k,:,j)&
                                           +resid(j,k,:,i) - resid(j,i,:,k)&
                                           +resid(k,i,:,j) - resid(k,j,:,i)

                           resid(i,k,:,j) = -resid(i,j,:,k)
                           resid(j,k,:,i) = resid(i,j,:,k)
                           resid(j,i,:,k) = -resid(i,j,:,k)
                           resid(k,i,:,j) = resid(i,j,:,k)
                           resid(k,j,:,i) = -resid(i,j,:,k)
                        end do
                     end do
                  end do
                  do i = 1,no
                     resid(i,i,:,:) = 0.0d0
                     resid(:,i,:,i) = 0.0d0
                     resid(i,:,:,i) = 0.0d0
                  end do


              end subroutine build_hr2

              subroutine build_hr4_p(resid,&
                                     r4_amps,r4_excits,&
                                     t2,r2,&
                                     h1_oo,h1_vv,&
                                     h2_vvov,h2_vooo,x2_oooo,x2_oovv,&
                                     h2_oooo,h2_voov,h2_vvvv,&
                                     n4,& 
                                     no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: t2(nu,nu,no,no),r2(no,no,nu,no),&
                                              h1_oo(no,no),h1_vv(nu,nu),&
                                              h2_vvov(nu,nu,no,nu),&
                                              h2_vooo(nu,no,no,no),&
                                              x2_oooo(no,no,no,no),&
                                              x2_oovv(no,no,nu,nu),&
                                              h2_oooo(no,no,no,no),&
                                              h2_voov(nu,no,no,nu),&
                                              h2_vvvv(nu,nu,nu,nu)

                  integer, intent(inout) :: r4_excits(n4,6)
                  !f2py intent(in,out) :: r4_excits(0:n4-1,0:5)
                  real(kind=8), intent(inout) :: r4_amps(n4)
                  !f2py intent(in,out) :: r4_amps(0:n4-1)

                  real(kind=8), intent(out) :: resid(n4)

                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: idx_table5(:,:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8), allocatable :: amps_buff(:), xbuf(:,:,:,:)
                  integer, allocatable :: excits_buff(:,:)

                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet
                  integer :: idx, nloc
                  real :: ONE, HALF
                  integer :: kout

                  HALF = (1.0d0/3.0d0) ! reduces error weridly well
                  !HALF = 1.0d0
                  ONE = 1.0d0

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: A(cd) h1(de)*r4(ijcekl)
                  !!!! diagram 3: 1/2 h2(cdef)*r4(ijefkl)
                  ! allocate new sorting arrays
                  nloc = no*(no - 1)*(no - 2)*(no - 3)/24
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(no,no,no,no))
                  !!! SB: (3,4,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-1,no-2/), (/-1,no-1/), (/-1,no/), no, no, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,4,5,6/), no, no, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_vv,h2_vvvv,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); d = r4_excits(jdet,2);
                        ! compute < abijkl | h2(vvvv) | cdijkl >
                        !hmatel = h2_vvvv(a,b,c,d)
                        hmatel = h2_vvvv(d,c,b,a) ! reorder for cache efficiency; (d,c) changes faster than (b,a)
                        ! compute < abijkl | h1(vv) | cdijkl >
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (a==c) hmatel1 = h1_vv(b,d)   ! (1)      < abijkl | h1(vv) | adijkl > 
                        if (a==d) hmatel2 = -h1_vv(b,c)  ! (cd)     < abijkl | h1(vv) | caijkl > 
                        if (b==c) hmatel3 = -h1_vv(a,d)  ! (ab)     < abijkl | h1(vv) | bdijkl >
                        if (b==d) hmatel4 = h1_vv(a,c)   ! (ab)(cd) < abijkl | h1(vv) | cdijkl >
                        hmatel = hmatel + (hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: -A(i/jkl) h1(mi)*r4(cdmjkl)
                  !!!! diagram 4: 1/2 A(ij/kl) h2(mnij)*r4(cdmnkl)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2  
                  ! allocate new sorting arrays
                  nloc = nu*(nu - 1)/2 * (no - 2)*(no - 3)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nu,nu,no,no))
                  !!! SB: (1,2,5,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/3,no-1/), (/-1,no/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,5,6/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnkl >
                        hmatel = h2_oooo(m,n,i,j)
                        ! compute < abijkl | h1(oo) | abmnkl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (i==m) hmatel1 = -h1_oo(n,j) ! (1)  
                        !if (i==n) hmatel2 = h1_oo(m,j)  ! (mn)
                        !if (j==m) hmatel3 = h1_oo(n,i)  ! (ij)
                        !if (j==n) hmatel4 = -h1_oo(m,i) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,i,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnil >
                        hmatel = -h2_oooo(m,n,k,j)
                        ! compute < abijkl | h1(oo) | abmnil >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==m) hmatel1 = h1_oo(n,j) ! (1)  
                        !if (k==n) hmatel2 = -h1_oo(m,j)  ! (mn)
                        !if (j==m) hmatel3 = -h1_oo(n,k)  ! (ij)
                        !if (j==n) hmatel4 = h1_oo(m,k) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnjl >
                        hmatel = -h2_oooo(m,n,i,k)
                        ! compute < abijkl | h1(oo) | abmnjl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (i==m) hmatel1 = h1_oo(n,k) ! (1)  
                        !if (i==n) hmatel2 = -h1_oo(m,k)  ! (mn)
                        !if (k==m) hmatel3 = -h1_oo(n,i)  ! (ij)
                        !if (k==n) hmatel4 = h1_oo(m,i) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il), -
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnik >
                        hmatel = h2_oooo(m,n,l,j)
                        ! compute < abijkl | h1(oo) | abmnik > = -< abijkl | abimkn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==m) hmatel1 = -h1_oo(n,j) ! (1)  
                        !if (l==n) hmatel2 = h1_oo(m,j)  ! (mn)
                        !if (j==m) hmatel3 = h1_oo(n,l)  ! (ij)
                        !if (j==n) hmatel4 = -h1_oo(m,l) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnjk >
                        hmatel = h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abmnjk >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (i==m) hmatel1 = -h1_oo(n,l) ! (1)  
                        !if (i==n) hmatel2 = h1_oo(m,l)  ! (mn)
                        !if (l==m) hmatel3 = h1_oo(n,i)  ! (ij)
                        !if (l==n) hmatel4 = -h1_oo(m,i) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(jl)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,4);
                        ! compute < abijkl | h2(oooo) | abmnij >
                        hmatel = h2_oooo(m,n,k,l)
                        ! compute < abijkl | h1(oo) | abmnij >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==m) hmatel1 = -h1_oo(n,l) ! (1)  
                        !if (k==n) hmatel2 = h1_oo(m,l)  ! (mn)
                        !if (l==m) hmatel3 = h1_oo(n,k)  ! (ij)
                        !if (l==n) hmatel4 = -h1_oo(m,k) ! (mn)(ij)
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-3,no/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,3,6/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abimnl >
                        hmatel = h2_oooo(m,n,j,k)
                        ! compute < abijkl | h1(oo) | abimnl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==j) hmatel1 = -h1_oo(n,k) ! (1)  
                        !if (n==j) hmatel2 = h1_oo(m,k)  ! (mn) 
                        !if (m==k) hmatel3 = h1_oo(n,j)  ! (jk) 
                        !if (n==k) hmatel4 = -h1_oo(m,j) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abkmnl >
                        hmatel = -h2_oooo(m,n,j,i)
                        ! compute < abijkl | h1(oo) | abkmnl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==j) hmatel1 = h1_oo(n,i) ! (1)  
                        !if (n==j) hmatel2 = -h1_oo(m,i)  ! (mn) 
                        !if (m==i) hmatel3 = -h1_oo(n,j)  ! (jk) 
                        !if (n==i) hmatel4 = h1_oo(m,j) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if 
                     ! (ij)
                     idx = idx_table(a,b,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abjmnl >
                        hmatel = -h2_oooo(m,n,i,k)
                        ! compute < abijkl | h1(oo) | abjmnl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==i) hmatel1 = h1_oo(n,k) ! (1)  
                        !if (n==i) hmatel2 = -h1_oo(m,k)  ! (mn) 
                        !if (m==k) hmatel3 = -h1_oo(n,i)  ! (jk) 
                        !if (n==k) hmatel4 = h1_oo(m,i) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if 
                     ! (kl)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abimnk >
                        hmatel = -h2_oooo(m,n,j,l)
                        ! compute < abijkl | h1(oo) | abimnk >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==j) hmatel1 = h1_oo(n,l) ! (1)  
                        !if (n==j) hmatel2 = -h1_oo(m,l)  ! (mn) 
                        !if (m==l) hmatel3 = -h1_oo(n,j)  ! (jk) 
                        !if (n==l) hmatel4 = h1_oo(m,j) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if 
                     ! (jl)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abimnj >
                        hmatel = -h2_oooo(m,n,l,k)
                        ! compute < abijkl | h1(oo) | abimnj >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==l) hmatel1 = h1_oo(n,k) ! (1)  
                        !if (n==l) hmatel2 = -h1_oo(m,k)  ! (mn) 
                        !if (m==k) hmatel3 = -h1_oo(n,l)  ! (jk) 
                        !if (n==k) hmatel4 = h1_oo(m,l) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if 
                     ! (ij)(kl)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abjmnk >
                        hmatel = h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abjmnk >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==i) hmatel1 = -h1_oo(n,l) ! (1)  
                        !if (n==i) hmatel2 = h1_oo(m,l)  ! (mn) 
                        !if (m==l) hmatel3 = h1_oo(n,i)  ! (jk) 
                        !if (n==l) hmatel4 = -h1_oo(m,i) ! (mn)(jk)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,6) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/2,no-2/), (/-2,no/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,4,6/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abmjnl >
                        hmatel = h2_oooo(m,n,i,k)
                        ! compute < abijkl | h1(oo) | abmjnl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==i) hmatel1 = -h1_oo(n,k) ! (1)  
                        !if (n==i) hmatel2 = h1_oo(m,k) ! (mn) 
                        !if (m==k) hmatel3 = h1_oo(n,i) ! (ik)  
                        !if (n==k) hmatel4 = -h1_oo(m,i) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abminl >
                        hmatel = -h2_oooo(m,n,j,k)
                        ! compute < abijkl | h1(oo) | abminl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==j) hmatel1 = h1_oo(n,k) ! (1)  
                        !if (n==j) hmatel2 = -h1_oo(m,k) ! (mn) 
                        !if (m==k) hmatel3 = -h1_oo(n,j) ! (ik)  
                        !if (n==k) hmatel4 = h1_oo(m,j) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abmjnk >
                        hmatel = -h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abmjnk >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==i) hmatel1 = h1_oo(n,l) ! (1)  
                        !if (n==i) hmatel2 = -h1_oo(m,l) ! (mn) 
                        !if (m==l) hmatel3 = -h1_oo(n,i) ! (ik)  
                        !if (n==l) hmatel4 = h1_oo(m,i) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il), -
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abminj >
                        hmatel = h2_oooo(m,n,l,k)
                        ! compute < abijkl | h2(oooo) | abminj >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==l) hmatel1 = -h1_oo(n,k) ! (1)  
                        !if (n==l) hmatel2 = h1_oo(m,k) ! (mn) 
                        !if (m==k) hmatel3 = h1_oo(n,l) ! (ik)  
                        !if (n==k) hmatel4 = -h1_oo(m,l) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abmknl >
                        hmatel = -h2_oooo(m,n,i,j)
                        ! compute < abijkl | h1(oo) | abmknl >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==i) hmatel1 = h1_oo(n,j) ! (1)  
                        !if (n==i) hmatel2 = -h1_oo(m,j) ! (mn) 
                        !if (m==j) hmatel3 = -h1_oo(n,i) ! (ik)  
                        !if (n==j) hmatel4 = h1_oo(m,i) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(kl)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,5);
                        ! compute < abijkl | h2(oooo) | abmink >
                        hmatel = h2_oooo(m,n,j,l)
                        ! compute < abijkl | h1(oo) | abmink >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (m==j) hmatel1 = -h1_oo(n,l) ! (1)  
                        !if (n==j) hmatel2 = h1_oo(m,l) ! (mn) 
                        !if (m==l) hmatel3 = h1_oo(n,j) ! (ik)  
                        !if (n==l) hmatel4 = -h1_oo(m,j) ! (mn)(ik) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-2,no-1/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,3,5/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abimkn >
                        hmatel = h2_oooo(m,n,j,l)
                        ! compute < abijkl | h1(oo) | abimkn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = -h1_oo(m,j) ! (1)  
                        !if (l==m) hmatel2 = h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = h1_oo(m,l) ! (jl) 
                        !if (j==m) hmatel4 = -h1_oo(n,l) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abjmkn >
                        hmatel = -h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abjmkn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = h1_oo(m,i) ! (1)  
                        !if (l==m) hmatel2 = -h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = -h1_oo(m,l) ! (jl) 
                        !if (i==m) hmatel4 = h1_oo(n,l) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,b,i,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abimln >
                        hmatel = -h2_oooo(m,n,j,k)
                        ! compute < abijkl | h1(oo) | abimln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==n) hmatel1 = h1_oo(m,j) ! (1)  
                        !if (k==m) hmatel2 = -h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = -h1_oo(m,k) ! (jl) 
                        !if (j==m) hmatel4 = h1_oo(n,k) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il), -
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abkmln >
                        hmatel = h2_oooo(m,n,j,i)
                        ! compute < abijkl | h1(oo) | abkmln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (i==n) hmatel1 = -h1_oo(m,j) ! (1)  
                        !if (i==m) hmatel2 = h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = h1_oo(m,i) ! (jl) 
                        !if (j==m) hmatel4 = -h1_oo(n,i) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abimjn >
                        hmatel = -h2_oooo(m,n,k,l)
                        ! compute < abijkl | h1(oo) | abimjn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = h1_oo(m,k) ! (1)  
                        !if (l==m) hmatel2 = -h1_oo(n,k) ! (mn) 
                        !if (k==n) hmatel3 = -h1_oo(m,l) ! (jl) 
                        !if (k==m) hmatel4 = h1_oo(n,l) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(kl)
                     idx = idx_table(a,b,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abjmln >
                        hmatel = h2_oooo(m,n,i,k)
                        ! compute < abijkl | h1(oo) | abjmln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==n) hmatel1 = -h1_oo(m,i) ! (1)  
                        !if (k==m) hmatel2 = h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = h1_oo(m,k) ! (jl) 
                        !if (i==m) hmatel4 = -h1_oo(n,k) ! (mn)(jl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,4,5) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/2,no-2/), (/-1,no-1/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,4,5/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmjkn >
                        hmatel = h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abmjkn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = -h1_oo(m,i) ! (1)  
                        !if (l==m) hmatel2 = h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = h1_oo(m,l) ! (il)  
                        !if (i==m) hmatel4 = -h1_oo(n,l) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmikn >
                        hmatel = -h2_oooo(m,n,j,l)
                        ! compute < abijkl | h1(oo) | abmikn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = h1_oo(m,j) ! (1)  
                        !if (l==m) hmatel2 = -h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = -h1_oo(m,l) ! (il)  
                        !if (j==m) hmatel4 = h1_oo(n,l) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmkln >
                        hmatel = h2_oooo(m,n,i,j)
                        ! compute < abijkl | h1(oo) | abmkln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (j==n) hmatel1 = -h1_oo(m,i) ! (1)  
                        !if (j==m) hmatel2 = h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = h1_oo(m,j) ! (il)  
                        !if (i==m) hmatel4 = -h1_oo(n,j) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmijn >
                        hmatel = h2_oooo(m,n,k,l)
                        ! compute < abijkl | h1(oo) | abmijn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = -h1_oo(m,k) ! (1)  
                        !if (l==m) hmatel2 = h1_oo(n,k) ! (mn) 
                        !if (k==n) hmatel3 = h1_oo(m,l) ! (il)  
                        !if (k==m) hmatel4 = -h1_oo(n,l) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(a,b,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmkln >
                        hmatel = -h2_oooo(m,n,i,k)
                        ! compute < abijkl | h1(oo) | abmkln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==n) hmatel1 = h1_oo(m,i) ! (1)  
                        !if (k==m) hmatel2 = -h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = -h1_oo(m,k) ! (il)  
                        !if (i==m) hmatel4 = h1_oo(n,k) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(kl)
                     idx = idx_table(a,b,i,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abmiln >
                        hmatel = h2_oooo(m,n,j,k)
                        ! compute < abijkl | h1(oo) | abmiln >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (k==n) hmatel1 = -h1_oo(m,j) ! (1)  
                        !if (k==m) hmatel2 = h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = h1_oo(m,k) ! (il)  
                        !if (j==m) hmatel4 = -h1_oo(n,k) ! (mn)(il)  
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4) LOOP !!!
                  call get_index_table(idx_table, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-1,no-2/), nu, nu, no, no)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/1,2,3,4/), nu, nu, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abijmn >
                        hmatel = h2_oooo(m,n,k,l)
                        ! compute < abijkl | h1(oo) | abijmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = -h1_oo(m,k) ! (1)  
                        !if (l==m) hmatel2 = h1_oo(n,k) ! (mn) 
                        !if (k==n) hmatel3 = h1_oo(m,l) ! (kl) 
                        !if (k==m) hmatel4 = -h1_oo(n,l) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik), -
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abjkmn >
                        hmatel = h2_oooo(m,n,i,l)
                        ! compute < abijkl | h1(oo) | abjkmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = -h1_oo(m,i) ! (1)  
                        !if (l==m) hmatel2 = h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = h1_oo(m,l) ! (kl) 
                        !if (i==m) hmatel4 = -h1_oo(n,l) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il), -
                     idx = idx_table(a,b,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abjlmn >
                        hmatel = h2_oooo(m,n,k,i)
                        ! compute < abijkl | h1(oo) | abjlmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (i==n) hmatel1 = -h1_oo(m,k) ! (1)  
                        !if (i==m) hmatel2 = h1_oo(n,k) ! (mn) 
                        !if (k==n) hmatel3 = h1_oo(m,i) ! (kl) 
                        !if (k==m) hmatel4 = -h1_oo(n,i) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abikmn >
                        hmatel = -h2_oooo(m,n,j,l)
                        ! compute < abijkl | h1(oo) | abikmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (l==n) hmatel1 = h1_oo(m,j) ! (1)  
                        !if (l==m) hmatel2 = -h1_oo(n,j) ! (mn) 
                        !if (j==n) hmatel3 = -h1_oo(m,l) ! (kl) 
                        !if (j==m) hmatel4 = h1_oo(n,l) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl)
                     idx = idx_table(a,b,i,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abilmn >
                        hmatel = -h2_oooo(m,n,k,j)
                        ! compute < abijkl | h1(oo) | abilmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (j==n) hmatel1 = h1_oo(m,k) ! (1)  
                        !if (j==m) hmatel2 = -h1_oo(n,k) ! (mn) 
                        !if (k==n) hmatel3 = -h1_oo(m,j) ! (kl) 
                        !if (k==m) hmatel4 = h1_oo(n,j) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(jl)
                     idx = idx_table(a,b,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(oooo) | abklmn >
                        hmatel = h2_oooo(m,n,i,j)
                        ! compute < abijkl | h1(oo) | abklmn >
                        !hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        !if (j==n) hmatel1 = -h1_oo(m,i) ! (1)  
                        !if (j==m) hmatel2 = h1_oo(n,i) ! (mn) 
                        !if (i==n) hmatel3 = h1_oo(m,j) ! (kl) 
                        !if (i==m) hmatel4 = -h1_oo(n,j) ! (mn)(kl) 
                        !hmatel = hmatel + HALF*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: -A(i/jkl) h1(mi)*r4(cdmjkl)
                  ! allocate new sorting arrays
                  nloc = nu*(nu - 1)/2 * (no - 1)*(no - 2)*(no - 3)/6
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table5(nu,nu,no,no,no))
                  !!! SB: (1,2,4,5,6) LOOP !!!
                  call get_index_table5(idx_table5, (/1,nu-1/), (/-1,nu/), (/2,no-2/), (/-1,no-1/), (/-1,no/), nu, nu, no, no, no)
                  call sort5(r4_excits, r4_amps, loc_arr, idx_table5, (/1,2,4,5,6/), nu, nu, no, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table5,&
                  !$omp h1_oo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table5(a,b,j,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3)
                        ! compute < abijkl | h1(oo) | abmjkl >
                        hmatel = -h1_oo(m,i)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table5(a,b,i,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3)
                        ! compute < abijkl | h1(oo) | abmikl >
                        hmatel = h1_oo(m,j)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table5(a,b,i,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3)
                        ! compute < abijkl | h1(oo) | abmijl >
                        hmatel = -h1_oo(m,k)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)
                     idx = idx_table5(a,b,i,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,3)
                        ! compute < abijkl | h1(oo) | abmijk >
                        hmatel = h1_oo(m,l)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,5,6) LOOP !!!
                  call get_index_table5(idx_table5, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-2,no-1/), (/-1,no/), nu, nu, no, no, no)
                  call sort5(r4_excits, r4_amps, loc_arr, idx_table5, (/1,2,3,5,6/), nu, nu, no, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table5,&
                  !$omp h1_oo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table5(a,b,i,k,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4)
                        ! compute < abijkl | h1(oo) | abimkl >
                        hmatel = -h1_oo(m,j)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table5(a,b,j,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4)
                        ! compute < abijkl | h1(oo) | abjmkl >
                        hmatel = h1_oo(m,i)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table5(a,b,i,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4)
                        ! compute < abijkl | h1(oo) | abimjl >
                        hmatel = h1_oo(m,k)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table5(a,b,i,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,4)
                        ! compute < abijkl | h1(oo) | abimjk >
                        hmatel = -h1_oo(m,l)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4,6) LOOP !!!
                  call get_index_table5(idx_table5, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-1,no-2/), (/-2,no/), nu, nu, no, no, no)
                  call sort5(r4_excits, r4_amps, loc_arr, idx_table5, (/1,2,3,4,6/), nu, nu, no, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table5,&
                  !$omp h1_oo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table5(a,b,i,j,l)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5)
                        ! compute < abijkl | h1(oo) | abijml >
                        hmatel = -h1_oo(m,k)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik), -
                     idx = idx_table5(a,b,j,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5)
                        ! compute < abijkl | h1(oo) | abjkml >
                        hmatel = -h1_oo(m,i)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table5(a,b,i,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5)
                        ! compute < abijkl | h1(oo) | abikml >
                        hmatel = h1_oo(m,j)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table5(a,b,i,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,5)
                        ! compute < abijkl | h1(oo) | abijmk >
                        hmatel = h1_oo(m,l)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (1,2,3,4,5) LOOP !!!
                  call get_index_table5(idx_table5, (/1,nu-1/), (/-1,nu/), (/1,no-3/), (/-1,no-2/), (/-1,no-1/), nu, nu, no, no, no)
                  call sort5(r4_excits, r4_amps, loc_arr, idx_table5, (/1,2,3,4,5/), nu, nu, no, no, no, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table5,&
                  !$omp h1_oo,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table5(a,b,i,j,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,6)
                        ! compute < abijkl | h1(oo) | abijkm >
                        hmatel = -h1_oo(m,l)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (il)
                     idx = idx_table5(a,b,j,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,6)
                        ! compute < abijkl | h1(oo) | abjklm >
                        hmatel = h1_oo(m,i)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table5(a,b,i,k,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,6)
                        ! compute < abijkl | h1(oo) | abiklm >
                        hmatel = -h1_oo(m,j)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table5(a,b,i,j,l)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = r4_excits(jdet,6)
                        ! compute < abijkl | h1(oo) | abijlm >
                        hmatel = h1_oo(m,k)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  deallocate(idx_table5,loc_arr)

                  !!!! diagram 5: A(cd)A(i/ijk) h2(dmle)*r4(ijcekm)
                  ! allocate new sorting arrays
                  nloc = (no - 1)*(no - 2)*(no - 3)/6 * (nu - 1) 
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(no,no,no,nu))
                  !!! SB: (3,4,5,1) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-1,no-2/), (/-1,no-1/), (/1,nu-1/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,4,5,1/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | adijkn >
                        hmatel = h2_voov(b,n,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (il)
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | adjkln >
                        hmatel = -h2_voov(b,n,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | adikln >
                        hmatel = h2_voov(b,n,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | adijln >
                        hmatel = -h2_voov(b,n,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | bdijkn >
                        hmatel = -h2_voov(a,n,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)(ab)
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | bdjkln >
                        hmatel = h2_voov(a,n,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl)(ab), -
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | bdikln >
                        hmatel = -h2_voov(a,n,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)(ab)
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | bdijln >
                        hmatel = h2_voov(a,n,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,6,1) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-1,no-2/), (/-2,no/), (/1,nu-1/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,4,6,1/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | adijml >
                        hmatel = h2_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik), -
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | adjkml >
                        hmatel = h2_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | adikml >
                        hmatel = -h2_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | adijmk >
                        hmatel = -h2_voov(b,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | bdijml >
                        hmatel = -h2_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(ab), -
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | bdjkml >
                        hmatel = -h2_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)(ab)
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | bdikml >
                        hmatel = h2_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)(ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | bdijmk >
                        hmatel = h2_voov(a,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,5,6,1) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-2,no-1/), (/-1,no/), (/1,nu-1/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,5,6,1/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,k,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | adimkl >
                        hmatel = h2_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | adjmkl >
                        hmatel = -h2_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | adimjl >
                        hmatel = -h2_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | adimjk >
                        hmatel = h2_voov(b,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | bdimkl >
                        hmatel = -h2_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | bdjmkl >
                        hmatel = h2_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)(ab)
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | bdimjl >
                        hmatel = h2_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl)(ab), -
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | bdimjk >
                        hmatel = -h2_voov(a,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,1) LOOP !!!
                  call get_index_table(idx_table, (/2,no-2/), (/-1,no-1/), (/-1,no/), (/1,nu-1/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/4,5,6,1/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,l,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | admjkl >
                        hmatel = h2_voov(b,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | admikl >
                        hmatel = -h2_voov(b,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | admijl >
                        hmatel = h2_voov(b,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | admijk >
                        hmatel = -h2_voov(b,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | bdmjkl >
                        hmatel = -h2_voov(a,m,i,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | bdmikl >
                        hmatel = h2_voov(a,m,j,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(ab), -
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | bdmijl >
                        hmatel = -h2_voov(a,m,k,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)(ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = r4_excits(jdet,2); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | bdmijk >
                        hmatel = h2_voov(a,m,l,d)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,5,2) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-1,no-2/), (/-1,no-1/), (/2,nu/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,4,5,2/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | cbijkn >
                        hmatel = h2_voov(a,n,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (il)
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | cbjkln >
                        hmatel = -h2_voov(a,n,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | cbikln >
                        hmatel = h2_voov(a,n,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | cbijln >
                        hmatel = -h2_voov(a,n,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | caijkn >
                        hmatel = -h2_voov(b,n,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)(ab)
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | cajkln >
                        hmatel = h2_voov(b,n,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl)(ab), -
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | caikln >
                        hmatel = -h2_voov(b,n,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)(ab)
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); n = r4_excits(jdet,6);
                        ! compute < abijkl | h2(voov) | caijln >
                        hmatel = h2_voov(b,n,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,4,6,2) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-1,no-2/), (/-2,no/), (/2,nu/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,4,6,2/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | cbijml >
                        hmatel = h2_voov(a,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ik), -
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | cbjkml >
                        hmatel = h2_voov(a,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | cbikml >
                        hmatel = -h2_voov(a,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | cbijmk >
                        hmatel = -h2_voov(a,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | caijml >
                        hmatel = -h2_voov(b,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(ab), -
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | cajkml >
                        hmatel = -h2_voov(b,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)(ab)
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | caikml >
                        hmatel = h2_voov(b,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (kl)(ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,5);
                        ! compute < abijkl | h2(voov) | caijmk >
                        hmatel = h2_voov(b,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (3,5,6,2) LOOP !!!
                  call get_index_table(idx_table, (/1,no-3/), (/-2,no-1/), (/-1,no/), (/2,nu/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/3,5,6,2/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,k,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | cbimkl >
                        hmatel = h2_voov(a,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(j,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | cbjmkl >
                        hmatel = -h2_voov(a,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | cbimjl >
                        hmatel = -h2_voov(a,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl), -
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | cbimjk >
                        hmatel = h2_voov(a,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | caimkl >
                        hmatel = -h2_voov(b,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | cajmkl >
                        hmatel = h2_voov(b,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jk)(ab)
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | caimjl >
                        hmatel = h2_voov(b,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (jl)(ab), -
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,4);
                        ! compute < abijkl | h2(voov) | caimjk >
                        hmatel = -h2_voov(b,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! SB: (4,5,6,2) LOOP !!!
                  call get_index_table(idx_table, (/2,no-2/), (/-1,no-1/), (/-1,no/), (/2,nu/), no, no, no, nu)
                  call sort4(r4_excits, r4_amps, loc_arr, idx_table, (/4,5,6,2/), no, no, no, nu, nloc, n4, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r4_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n4),&
                  !$omp private(hmatel,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     idx = idx_table(j,k,l,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | cbmjkl >
                        hmatel = h2_voov(a,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(i,k,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | cbmikl >
                        hmatel = -h2_voov(a,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik), -
                     idx = idx_table(i,j,l,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | cbmijl >
                        hmatel = h2_voov(a,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | cbmijk >
                        hmatel = -h2_voov(a,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(j,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | camjkl >
                        hmatel = -h2_voov(b,m,i,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ij)(ab)
                     idx = idx_table(i,k,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | camikl >
                        hmatel = h2_voov(b,m,j,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (ik)(ab), -
                     idx = idx_table(i,j,l,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | camijl >
                        hmatel = -h2_voov(b,m,k,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                     ! (il)(ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        c = r4_excits(jdet,1); m = r4_excits(jdet,3);
                        ! compute < abijkl | h2(voov) | camijk >
                        hmatel = h2_voov(b,m,l,c)
                        resid(idet) = resid(idet) + hmatel*r4_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  deallocate(idx_table,loc_arr)

                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp r4_excits,&
                  !$omp r2,t2,&
                  !$omp h2_vvov,h2_vooo,&
                  !$omp x2_oovv,x2_oooo,&
                  !$omp no,nu,n4),&
                  !$omp private(idet,a,b,c,d,i,j,k,l,m,n,e,f,&
                  !$omp res_mm23)
                  !$omp do schedule(static)
                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);

                     res_mm23 = 0.0d0
                     do m = 1,no
                        ! A(ij/kl)A(ab) -h2(bmlk)*r2(ijam)
                        ! (1)
                        res_mm23 = res_mm23 - h2_vooo(b,m,l,k)*r2(i,j,a,m) ! (1) 
                        res_mm23 = res_mm23 + h2_vooo(b,m,l,i)*r2(k,j,a,m) ! (ik) 
                        res_mm23 = res_mm23 + h2_vooo(b,m,i,k)*r2(l,j,a,m) ! (il) 
                        res_mm23 = res_mm23 + h2_vooo(b,m,l,j)*r2(i,k,a,m) ! (jk) 
                        res_mm23 = res_mm23 + h2_vooo(b,m,j,k)*r2(i,l,a,m) ! (jl) 
                        res_mm23 = res_mm23 - h2_vooo(b,m,j,i)*r2(k,l,a,m) ! (ik)(jl) 
                        ! (ab)
                        res_mm23 = res_mm23 + h2_vooo(a,m,l,k)*r2(i,j,b,m) ! (1) 
                        res_mm23 = res_mm23 - h2_vooo(a,m,l,i)*r2(k,j,b,m) ! (ik) 
                        res_mm23 = res_mm23 - h2_vooo(a,m,i,k)*r2(l,j,b,m) ! (il) 
                        res_mm23 = res_mm23 - h2_vooo(a,m,l,j)*r2(i,k,b,m) ! (jk) 
                        res_mm23 = res_mm23 - h2_vooo(a,m,j,k)*r2(i,l,b,m) ! (jl) 
                        res_mm23 = res_mm23 + h2_vooo(a,m,j,i)*r2(k,l,b,m) ! (ik)(jl) 
                        ! A(l/ijk) -x2(ijmk)*t2(abml)
                        res_mm23 = res_mm23 - x2_oooo(i,j,m,k)*t2(a,b,m,l) ! (1)
                        res_mm23 = res_mm23 + x2_oooo(l,j,m,k)*t2(a,b,m,i) ! (il)
                        res_mm23 = res_mm23 + x2_oooo(i,l,m,k)*t2(a,b,m,j) ! (jl)
                        res_mm23 = res_mm23 + x2_oooo(i,j,m,l)*t2(a,b,m,k) ! (kl)
                     end do
                     do e = 1,nu
                        ! A(l/ijk) h2(bale)*r2(ijek)
                        res_mm23 = res_mm23 + h2_vvov(b,a,l,e)*r2(i,j,e,k) ! (1)
                        res_mm23 = res_mm23 - h2_vvov(b,a,i,e)*r2(l,j,e,k) ! (il)
                        res_mm23 = res_mm23 - h2_vvov(b,a,j,e)*r2(i,l,e,k) ! (jl)
                        res_mm23 = res_mm23 - h2_vvov(b,a,k,e)*r2(i,j,e,l) ! (kl)
                        ! A(ij/kl)A(ab) x2(ijae)*t2(ebkl)
                        ! (1)
                        res_mm23 = res_mm23 + x2_oovv(i,j,a,e)*t2(e,b,k,l) ! (1)
                        res_mm23 = res_mm23 - x2_oovv(k,j,a,e)*t2(e,b,i,l) ! (ik)
                        res_mm23 = res_mm23 - x2_oovv(l,j,a,e)*t2(e,b,k,i) ! (il)
                        res_mm23 = res_mm23 - x2_oovv(i,k,a,e)*t2(e,b,j,l) ! (jk)
                        res_mm23 = res_mm23 - x2_oovv(i,l,a,e)*t2(e,b,k,j) ! (jl)
                        res_mm23 = res_mm23 + x2_oovv(k,l,a,e)*t2(e,b,i,j) ! (ik)(jl)
                        ! (ab)
                        res_mm23 = res_mm23 - x2_oovv(i,j,b,e)*t2(e,a,k,l) ! (1)
                        res_mm23 = res_mm23 + x2_oovv(k,j,b,e)*t2(e,a,i,l) ! (ik)
                        res_mm23 = res_mm23 + x2_oovv(l,j,b,e)*t2(e,a,k,i) ! (il)
                        res_mm23 = res_mm23 + x2_oovv(i,k,b,e)*t2(e,a,j,l) ! (jk)
                        res_mm23 = res_mm23 + x2_oovv(i,l,b,e)*t2(e,a,k,j) ! (jl)
                        res_mm23 = res_mm23 - x2_oovv(k,l,b,e)*t2(e,a,i,j) ! (ik)(jl)
                     end do
                     resid(idet) = resid(idet) + res_mm23
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!

              end subroutine build_hr4_p

              subroutine build_I_oooo(I_oooo,&
                                      r4_amps,r4_excits,&
                                      h2_oovv,&
                                      n4,& 
                                      no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: h2_oovv(no,no,nu,nu)
                  integer, intent(in) :: r4_excits(n4,6)
                  real(kind=8), intent(in) :: r4_amps(n4)

                  real(kind=8), intent(inout) :: I_oooo(no,no,no,no)
                  !f2py intent(in,out) :: I_oooo(0:no-1,0:no-1,0:no-1,0:no-1)

                  real(kind=8) :: val, rval
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet

                  do idet = 1,n4
                     rval = r4_amps(idet)
                     ! I(ijmk) <- 1/2 g(mlab) * r3(abijkl)
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     I_oooo(i,j,:,k) = I_oooo(i,j,:,k) + h2_oovv(:,l,a,b)*rval ! (1)
                     I_oooo(j,k,:,l) = I_oooo(j,k,:,l) - h2_oovv(:,i,a,b)*rval ! (il)
                     I_oooo(i,k,:,l) = I_oooo(i,k,:,l) + h2_oovv(:,j,a,b)*rval ! (jl)
                     I_oooo(i,j,:,l) = I_oooo(i,j,:,l) - h2_oovv(:,k,a,b)*rval ! (kl)
                  end do

                  ! antisymmetrize A(ijk)
                  do i = 1,no
                     do j = i+1,no
                        do k = j+1,no
                           do m = 1,no
                              val =   I_oooo(i,j,m,k) - I_oooo(i,k,m,j)&
                                     +I_oooo(j,k,m,i) - I_oooo(j,i,m,k)&
                                     +I_oooo(k,i,m,j) - I_oooo(k,j,m,i)
                              I_oooo(i,j,m,k) = val
                              I_oooo(i,k,m,j) = -val
                              I_oooo(j,k,m,i) = val
                              I_oooo(j,i,m,k) = -val
                              I_oooo(k,i,m,j) = val
                              I_oooo(k,j,m,i) = -val
                           end do
                        end do
                     end do
                  end do
                  ! manually zero out diagonal elements
                  do i = 1,no
                     I_oooo(i,i,:,:) = 0.0d0
                     I_oooo(:,i,:,i) = 0.0d0
                     I_oooo(i,:,:,i) = 0.0d0
                  end do

              end subroutine build_I_oooo

              subroutine build_I_oovv(I_oovv,&
                                      r4_amps,r4_excits,&
                                      h2_oovv,&
                                      n4,& 
                                      no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: h2_oovv(no,no,nu,nu)
                  integer, intent(in) :: r4_excits(n4,6)
                  real(kind=8), intent(in) :: r4_amps(n4)

                  real(kind=8), intent(inout) :: I_oovv(no,no,nu,nu)
                  !f2py intent(in,out) :: I_oovv(0:no-1,0:no-1,0:nu-1,0:nu-1)

                  real(kind=8) :: val, rval
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet

                  do idet = 1,n4
                     rval = r4_amps(idet)
                     ! I(ijae) <- 1/2 A(ab)A(kl/ij) g(kleb) * r3(abijkl)
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);
                     ! (1)
                     I_oovv(i,j,a,:) = I_oovv(i,j,a,:) - h2_oovv(k,l,:,b)*rval ! (1)
                     I_oovv(j,k,a,:) = I_oovv(j,k,a,:) - h2_oovv(i,l,:,b)*rval ! (ik)
                     I_oovv(j,l,a,:) = I_oovv(j,l,a,:) - h2_oovv(k,i,:,b)*rval ! (il)
                     I_oovv(i,k,a,:) = I_oovv(i,k,a,:) + h2_oovv(j,l,:,b)*rval ! (jk)
                     I_oovv(i,l,a,:) = I_oovv(i,l,a,:) + h2_oovv(k,j,:,b)*rval ! (jl)
                     I_oovv(k,l,a,:) = I_oovv(k,l,a,:) - h2_oovv(i,j,:,b)*rval ! (ik)(jl)
                     ! (ab)
                     I_oovv(i,j,b,:) = I_oovv(i,j,b,:) + h2_oovv(k,l,:,a)*rval ! (1)
                     I_oovv(j,k,b,:) = I_oovv(j,k,b,:) + h2_oovv(i,l,:,a)*rval ! (ik)
                     I_oovv(j,l,b,:) = I_oovv(j,l,b,:) + h2_oovv(k,i,:,a)*rval ! (il)
                     I_oovv(i,k,b,:) = I_oovv(i,k,b,:) - h2_oovv(j,l,:,a)*rval ! (jk)
                     I_oovv(i,l,b,:) = I_oovv(i,l,b,:) - h2_oovv(k,j,:,a)*rval ! (jl)
                     I_oovv(k,l,b,:) = I_oovv(k,l,b,:) + h2_oovv(i,j,:,a)*rval ! (ik)(jl)
                  end do

                  ! antisymmetrize A(ij)
                  do i = 1,no
                     do j = i+1,no
                        do a = 1,nu
                           do e = 1,nu
                              val = I_oovv(i,j,a,e) - I_oovv(j,i,a,e)
                              I_oovv(i,j,a,e) = val
                              I_oovv(j,i,a,e) = -val
                           end do
                        end do
                     end do
                  end do
                  ! manually zero out diagonal elements
                  do i = 1,no
                     I_oovv(i,i,:,:) = 0.0d0
                  end do

              end subroutine build_I_oovv

              subroutine update_r(r1,r2,r4_amps,r4_excits,&
                                  omega,&
                                  h1_oo,h1_vv,&
                                  n4,no,nu)

                  integer, intent(in) :: no, nu, n4
                  real(kind=8), intent(in) :: h1_oo(no,no),h1_vv(nu,nu)
                  real(kind=8), intent(in) :: omega
                  integer, intent(in) :: r4_excits(n4,6)

                  real(kind=8), intent(inout) :: r1(no,no)
                  !f2py intent(in,out) :: r1(0:no-1,0:no-1)
                  real(kind=8), intent(inout) :: r2(no,no,nu,no)
                  !f2py intent(in,out) :: r2(0:no-1,0:no-1,0:nu-1,0:no-1)
                  real(kind=8), intent(inout) :: r4_amps(n4)
                  !f2py intent(in,out) :: r4_amps(0:n4-1)

                  integer :: idet, a, b, c, i, j, k, l
                  real(kind=8) :: denom

                  do i = 1,no
                     do j = 1,no
                        if (i==j) then
                           r1(i,j) = 0.0d0
                        end if
                        denom = omega + h1_oo(i,i) + h1_oo(j,j)
                        r1(i,j) = r1(i,j)/denom
                     end do
                  end do

                  do i = 1,no
                     do j = 1,no
                        do c = 1,nu
                           do k = 1,no
                              if (i==j .or. j==k .or. i==k) then
                                 r2(i,j,c,k) = 0.0d0
                              end if
                              denom = omega + h1_oo(i,i) + h1_oo(j,j) + h1_oo(k,k) - h1_vv(c,c)
                              r2(i,j,c,k) = r2(i,j,c,k)/denom
                           end do
                        end do
                     end do
                  end do

                  do idet = 1,n4
                     a = r4_excits(idet,1); b = r4_excits(idet,2); 
                     i = r4_excits(idet,3); j = r4_excits(idet,4); k = r4_excits(idet,5); l = r4_excits(idet,6);

                     denom = omega + h1_oo(i,i) + h1_oo(j,j) + h1_oo(k,k) + h1_oo(l,l) - h1_vv(a,a) - h1_vv(b,b)
                     r4_amps(idet) = r4_amps(idet)/denom
                  end do

              end subroutine update_r

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine get_index_table(idx_table, rng1, rng2, rng3, rng4, n1, n2, n3, n4)

              integer, intent(in) :: n1, n2, n3, n4
              integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2)

              integer, intent(inout) :: idx_table(n1,n2,n3,n4)

              integer :: kout
              integer :: p, q, r, s

              idx_table = 0
              ! 5 possible cases. Always organize so that ordered indices appear first.
              if (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) < 0) then ! p < q < r < s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = q-rng3(1), rng3(2)
                          do s = r-rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) > 0) then ! p < q < r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = q-rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) < 0) then ! p < q, r < s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = r-rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              elseif (rng1(1) > 0 .and. rng2(1) < 0 .and. rng3(1) > 0 .and. rng4(1) > 0) then ! p < q, r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = p-rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              else ! p, q, r, s
                 kout = 1 
                 do p = rng1(1), rng1(2)
                    do q = rng2(1), rng2(2)
                       do r = rng3(1), rng3(2)
                          do s = rng4(1), rng4(2)
                             idx_table(p,q,r,s) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              end if

      end subroutine get_index_table

      subroutine get_index_table5(idx_table, rng1, rng2, rng3, rng4, rng5, n1, n2, n3, n4, n5)

              integer, intent(in) :: n1, n2, n3, n4, n5
              integer, intent(in) :: rng1(2), rng2(2), rng3(2), rng4(2), rng5(2)

              integer, intent(inout) :: idx_table(n1,n2,n3,n4,n5)

              integer :: kout
              integer :: p, q, r, s, t

              idx_table = 0
              ! only do the p < q, r < s < t case because that is what we need
              kout = 1 
              do p = rng1(1), rng1(2)
                 do q = p-rng2(1), rng2(2)
                    do r = rng3(1), rng3(2)
                       do s = r-rng4(1), rng4(2)
                          do t = s-rng5(1), rng5(2)
                             idx_table(p,q,r,s,t) = kout
                             kout = kout + 1
                          end do
                       end do
                    end do
                 end do
              end do

      end subroutine get_index_table5

      subroutine sort4(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, nloc, n3p, resid)
      ! Sort the 1D array of T3 amplitudes, the 2D array of T3 excitations, and, optionally, the
      ! associated 1D residual array such that triple excitations with the same spatial orbital
      ! indices in the positions indicated by idims are next to one another.
      ! In:
      !   idims: array of 4 integer dimensions along which T3 will be sorted
      !   n1, n2, n3, and n4: no/nu sizes of each dimension in idims
      !   nloc: permutationally unique number of possible (p,q,r,s) tuples
      !   n3p: Number of P-space triples of interest
      ! In,Out:
      !   excits: T3 excitation array (can be aaa, aab, abb, or bbb)
      !   amps: T3 amplitude vector (can be aaa, aab, abb, or bbb)
      !   resid (optional): T3 residual vector (can be aaa, aab, abb, or bbb)
      !   loc_arr: array providing the start- and end-point indices for each sorted block in t3 excitations
          
              integer, intent(in) :: n1, n2, n3, n4, nloc, n3p
              integer, intent(in) :: idims(4)
              integer, intent(in) :: idx_table(n1,n2,n3,n4)

              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(n3p,6)
              real(kind=8), intent(inout) :: amps(n3p)
              real(kind=8), intent(inout), optional :: resid(n3p)

              integer :: idet
              integer :: p, q, r, s
              integer :: p1, q1, r1, s1, p2, q2, r2, s2
              integer :: pqrs1, pqrs2
              integer, allocatable :: temp(:), idx(:)

              ! obtain the lexcial index for each triple excitation in the P space along the sorting dimensions idims
              allocate(temp(n3p),idx(n3p))
              do idet = 1, n3p
                 p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3)); s = excits(idet,idims(4))
                 temp(idet) = idx_table(p,q,r,s)
              end do
              ! get the sorting array
              call argsort(temp, idx)
              ! apply sorting array to t3 excitations, amplitudes, and, optionally, residual arrays
              excits = excits(idx,:)
              amps = amps(idx)
              if (present(resid)) resid = resid(idx)
              deallocate(temp,idx)
              ! obtain the start- and end-point indices for each lexical index in the sorted t3 excitation and amplitude arrays
              loc_arr(1,:) = 1; loc_arr(2,:) = 0; ! set default start > end so that empty sets do not trigger loops
              !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
              !if (n3p <= 1) print*, "(ccsdt_p_loops) >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
              if (n3p == 1) then
                 if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and. excits(1,5)==1 .and. excits(1,6)==1) return
                 p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3)); s2 = excits(n3p,idims(4))
                 pqrs2 = idx_table(p2,q2,r2,s2)
              else               
                 pqrs2 = -1
              end if
              do idet = 1, n3p-1
                 ! get consecutive lexcial indices
                 p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));   s1 = excits(idet,idims(4))
                 p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3)); s2 = excits(idet+1,idims(4))
                 pqrs1 = idx_table(p1,q1,r1,s1)
                 pqrs2 = idx_table(p2,q2,r2,s2)
                 ! if change occurs between consecutive indices, record these locations in loc_arr as new start/end points
                 if (pqrs1 /= pqrs2) then
                    loc_arr(2,pqrs1) = idet
                    loc_arr(1,pqrs2) = idet+1
                 end if
              end do
              !if (n3p > 1) then
              loc_arr(2,pqrs2) = n3p
              !end if

      end subroutine sort4

      subroutine sort5(excits, amps, loc_arr, idx_table, idims, n1, n2, n3, n4, n5, nloc, n3p, resid)
          
              integer, intent(in) :: n1, n2, n3, n4, n5, nloc, n3p
              integer, intent(in) :: idims(5)
              integer, intent(in) :: idx_table(n1,n2,n3,n4,n5)

              integer, intent(inout) :: loc_arr(2,nloc)
              integer, intent(inout) :: excits(n3p,6)
              real(kind=8), intent(inout) :: amps(n3p)
              real(kind=8), intent(inout), optional :: resid(n3p)

              integer :: idet
              integer :: p, q, r, s, t
              integer :: p1, q1, r1, s1, t1, p2, q2, r2, s2, t2
              integer :: pqrst1, pqrst2
              integer, allocatable :: temp(:), idx(:)

              ! obtain the lexcial index for each triple excitation in the P space along the sorting dimensions idims
              allocate(temp(n3p),idx(n3p))
              do idet = 1, n3p
                 p = excits(idet,idims(1)); q = excits(idet,idims(2)); r = excits(idet,idims(3));
                 s = excits(idet,idims(4)); t = excits(idet,idims(5))
                 temp(idet) = idx_table(p,q,r,s,t)
              end do
              ! get the sorting array
              call argsort(temp, idx)
              ! apply sorting array to t3 excitations, amplitudes, and, optionally, residual arrays
              excits = excits(idx,:)
              amps = amps(idx)
              if (present(resid)) resid = resid(idx)
              deallocate(temp,idx)
              ! obtain the start- and end-point indices for each lexical index in the sorted t3 excitation and amplitude arrays
              loc_arr(1,:) = 1; loc_arr(2,:) = 0; ! set default start > end so that empty sets do not trigger loops
              !!! WARNING: THERE IS A MEMORY LEAK HERE! pqrs2 is used below but is not set if n3p <= 1
              !if (n3p <= 1) print*, "(ccsdt_p_loops) >> WARNING: potential memory leakage in sort4 function. pqrs2 set to -1"
              if (n3p == 1) then
                 if (excits(1,1)==1 .and. excits(1,2)==1 .and. excits(1,3)==1 .and. excits(1,4)==1 .and.&
                     excits(1,5)==1 .and. excits(1,6)==1) return
                 p2 = excits(n3p,idims(1)); q2 = excits(n3p,idims(2)); r2 = excits(n3p,idims(3));
                 s2 = excits(n3p,idims(4)); t2 = excits(n3p,idims(5));
                 pqrst2 = idx_table(p2,q2,r2,s2,t2)
              else               
                 pqrst2 = -1
              end if
              do idet = 1, n3p-1
                 ! get consecutive lexcial indices
                 p1 = excits(idet,idims(1));   q1 = excits(idet,idims(2));   r1 = excits(idet,idims(3));
                 s1 = excits(idet,idims(4)); t1 = excits(idet,idims(5));
                 p2 = excits(idet+1,idims(1)); q2 = excits(idet+1,idims(2)); r2 = excits(idet+1,idims(3));
                 s2 = excits(idet+1,idims(4)); t2 = excits(idet+1,idims(5));
                 pqrst1 = idx_table(p1,q1,r1,s1,t1)
                 pqrst2 = idx_table(p2,q2,r2,s2,t2)
                 ! if change occurs between consecutive indices, record these locations in loc_arr as new start/end points
                 if (pqrst1 /= pqrst2) then
                    loc_arr(2,pqrst1) = idet
                    loc_arr(1,pqrst2) = idet+1
                 end if
              end do
              !if (n3p > 1) then
              loc_arr(2,pqrst2) = n3p
              !end if

      end subroutine sort5

      subroutine argsort(r,d)

              integer, intent(in), dimension(:) :: r
              integer, intent(out), dimension(size(r)) :: d

              integer, dimension(size(r)) :: il

              integer :: stepsize
              integer :: i, j, n, left, k, ksize

              n = size(r)

              do i=1,n
                 d(i)=i
              end do

              if (n==1) return

              stepsize = 1
              do while (stepsize < n)
                 do left = 1, n-stepsize,stepsize*2
                    i = left
                    j = left+stepsize
                    ksize = min(stepsize*2,n-left+1)
                    k=1

                    do while (i < left+stepsize .and. j < left+ksize)
                       if (r(d(i)) < r(d(j))) then
                          il(k) = d(i)
                          i = i+1
                          k = k+1
                       else
                          il(k) = d(j)
                          j = j+1
                          k = k+1
                       endif
                    enddo

                    if (i < left+stepsize) then
                       ! fill up remaining from left
                       il(k:ksize) = d(i:left+stepsize-1)
                    else
                       ! fill up remaining from right
                       il(k:ksize) = d(j:left+ksize-1)
                    endif
                    d(left:left+ksize-1) = il(1:ksize)
                 end do
                 stepsize = stepsize*2
              end do

      end subroutine argsort
      
      subroutine reorder4(y, x, iorder)

          integer, intent(in) :: iorder(4)
          real(kind=8), intent(in) :: x(:,:,:,:)

          real(kind=8), intent(out) :: y(:,:,:,:)

          integer :: i, j, k, l
          integer :: vec(4)

          y = 0.0d0
          do i = 1, size(x,1)
             do j = 1, size(x,2)
                do k = 1, size(x,3)
                   do l = 1, size(x,4)
                      vec = (/i,j,k,l/)
                      y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4))) = x(i,j,k,l)
                   end do
                end do
             end do
          end do

      end subroutine reorder4
    
      subroutine sum4(x, y, iorder)

          integer, intent(in) :: iorder(4)
          real(kind=8), intent(in) :: y(:,:,:,:)

          real(kind=8), intent(inout) :: x(:,:,:,:)
          
          integer :: i, j, k, l
          integer :: vec(4)

          do i = 1, size(x,1)
             do j = 1, size(x,2)
                do k = 1, size(x,3)
                   do l = 1, size(x,4)
                      vec = (/i,j,k,l/)
                      x(i,j,k,l) = x(i,j,k,l) + y(vec(iorder(1)),vec(iorder(2)),vec(iorder(3)),vec(iorder(4)))
                   end do
                end do
             end do
          end do

      end subroutine sum4

end module dipeom4_p
