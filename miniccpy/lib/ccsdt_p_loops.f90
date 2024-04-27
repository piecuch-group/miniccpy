module ccsdt_p_loops
    
      use omp_lib

      implicit none

      contains

               subroutine update_t1(t1,resid,&
                                    singles_res,&
                                    t3_excits,t3_amps,&
                                    h2_oovv,&
                                    f_oo,f_vv,&
                                    shift,&
                                    n3,&
                                    no, nu)

                      integer, intent(in) :: no, nu, n3
                      integer, intent(in) :: t3_excits(n3,6)
                      real(kind=8), intent(in) :: t3_amps(n3)
                      real(kind=8), intent(in) :: singles_res(1:nu,1:no),&
                                                  h2_oovv(1:no,1:no,1:nu,1:nu),&
                                                  f_oo(1:no,1:no), f_vv(1:nu,1:nu),&
                                                  shift

                      real(kind=8), intent(inout) :: t1(1:nu,1:no)
                      !f2py intent(in,out) :: t1(0:nu-1,0:no-1)

                      real(kind=8), intent(out) :: resid(1:nu,1:no)

                      integer :: i, a, m, n, e, f, idet
                      real(kind=8) :: denom, val, t_amp

                      ! store x1a in resid container
                      resid(:,:) = singles_res(:,:)
                      ! compute < ia | (H(2) * T3)_C | 0 >
                      do idet = 1, n3
                          t_amp = t3_amps(idet)
                          ! A(a/ef)A(i/mn) h2(mnef) * t3(aefimn)
                          a = t3_excits(idet,1); e = t3_excits(idet,2); f = t3_excits(idet,3);
                          i = t3_excits(idet,4); m = t3_excits(idet,5); n = t3_excits(idet,6);
                          resid(a,i) = resid(a,i) + h2_oovv(m,n,e,f) * t_amp ! (1)
                          resid(e,i) = resid(e,i) - h2_oovv(m,n,a,f) * t_amp ! (ae)
                          resid(f,i) = resid(f,i) - h2_oovv(m,n,e,a) * t_amp ! (af)
                          resid(a,m) = resid(a,m) - h2_oovv(i,n,e,f) * t_amp ! (im)
                          resid(e,m) = resid(e,m) + h2_oovv(i,n,a,f) * t_amp ! (ae)(im)
                          resid(f,m) = resid(f,m) + h2_oovv(i,n,e,a) * t_amp ! (af)(im)
                          resid(a,n) = resid(a,n) - h2_oovv(m,i,e,f) * t_amp ! (in)
                          resid(e,n) = resid(e,n) + h2_oovv(m,i,a,f) * t_amp ! (ae)(in)
                          resid(f,n) = resid(f,n) + h2_oovv(m,i,e,a) * t_amp ! (af)(in)
                      end do
                      ! update loop
                      do i = 1,no
                          do a = 1,nu
                              denom = f_oo(i,i) - f_vv(a,a)
                              val = resid(a,i)/(denom - shift)
                              t1(a,i) = t1(a,i) + val
                              resid(a,i) = val
                          end do
                      end do

              end subroutine update_t1

              subroutine update_t2(t2,resid,&
                                   doubles_res,&
                                   t3_excits,t3_amps,&
                                   h1_ov,&
                                   h2_ooov, h2_vovv,&
                                   f_oo, f_vv,&
                                   shift,&
                                   n3,&
                                   no, nu)

                  integer, intent(in) :: no, nu, n3
                  integer, intent(in) :: t3_excits(n3,6)
                  real(kind=8), intent(in) :: t3_amps(n3)
                  real(kind=8), intent(in) :: doubles_res(1:nu,1:nu,1:no,1:no),&
                                              h1_ov(1:no,1:nu),&
                                              h2_ooov(1:no,1:no,1:no,1:nu),&
                                              h2_vovv(1:nu,1:no,1:nu,1:nu),&
                                              f_oo(1:no,1:no), f_vv(1:nu,1:nu),&
                                              shift

                  real(kind=8), intent(inout) :: t2(1:nu,1:nu,1:no,1:no)
                  !f2py intent(in,out) :: t2(0:nu-1,0:nu-1,0:no-1,0:no-1)

                  real(kind=8), intent(out) :: resid(1:nu,1:nu,1:no,1:no)

                  integer :: i, j, a, b, m, n, e, f, idet
                  real(kind=8) :: denom, val, t_amp

                  ! Store x2a in residual container
                  resid(:,:,:,:) = doubles_res(:,:,:,:)
                  ! compute < ijab | (H(2) * T3)_C | 0 >
                  do idet = 1, n3
                      t_amp = t3_amps(idet)

                      ! A(ij)A(ab) [A(m/ij)A(e/ab) h1a(me) * t3(abeijm)]
                      a = t3_excits(idet,1); b = t3_excits(idet,2); e = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); m = t3_excits(idet,6);
                      resid(a,b,i,j) = resid(a,b,i,j) + h1_ov(m,e) * t_amp ! (1)
                      resid(a,b,m,j) = resid(a,b,m,j) - h1_ov(i,e) * t_amp ! (im)
                      resid(a,b,i,m) = resid(a,b,i,m) - h1_ov(j,e) * t_amp ! (jm)
                      resid(e,b,i,j) = resid(e,b,i,j) - h1_ov(m,a) * t_amp ! (ae)
                      resid(e,b,m,j) = resid(e,b,m,j) + h1_ov(i,a) * t_amp ! (im)(ae)
                      resid(e,b,i,m) = resid(e,b,i,m) + h1_ov(j,a) * t_amp ! (jm)(ae)
                      resid(a,e,i,j) = resid(a,e,i,j) - h1_ov(m,b) * t_amp ! (be)
                      resid(a,e,m,j) = resid(a,e,m,j) + h1_ov(i,b) * t_amp ! (im)(be)
                      resid(a,e,i,m) = resid(a,e,i,m) + h1_ov(j,b) * t_amp ! (jm)(be)

                      ! A(ij)A(ab) [A(j/mn)A(f/ab) -h2(mnif) * t3(abfmjn)]
                      a = t3_excits(idet,1); b = t3_excits(idet,2); f = t3_excits(idet,3);
                      m = t3_excits(idet,4); j = t3_excits(idet,5); n = t3_excits(idet,6);
                      resid(a,b,:,j) = resid(a,b,:,j) - h2_ooov(m,n,:,f) * t_amp ! (1)
                      resid(a,b,:,m) = resid(a,b,:,m) + h2_ooov(j,n,:,f) * t_amp ! (jm)
                      resid(a,b,:,n) = resid(a,b,:,n) + h2_ooov(m,j,:,f) * t_amp ! (jn)
                      resid(f,b,:,j) = resid(f,b,:,j) + h2_ooov(m,n,:,a) * t_amp ! (af)
                      resid(f,b,:,m) = resid(f,b,:,m) - h2_ooov(j,n,:,a) * t_amp ! (jm)(af)
                      resid(f,b,:,n) = resid(f,b,:,n) - h2_ooov(m,j,:,a) * t_amp ! (jn)(af)
                      resid(a,f,:,j) = resid(a,f,:,j) + h2_ooov(m,n,:,b) * t_amp ! (bf)
                      resid(a,f,:,m) = resid(a,f,:,m) - h2_ooov(j,n,:,b) * t_amp ! (jm)(bf)
                      resid(a,f,:,n) = resid(a,f,:,n) - h2_ooov(m,j,:,b) * t_amp ! (jn)(bf)

                      ! A(ij)A(ab) [A(n/ij)A(b/ef) h2(anef) * t3(ebfijn)]
                      e = t3_excits(idet,1); b = t3_excits(idet,2); f = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); n = t3_excits(idet,6);
                      resid(:,b,i,j) = resid(:,b,i,j) + h2_vovv(:,n,e,f) * t_amp ! (1)
                      resid(:,b,n,j) = resid(:,b,n,j) - h2_vovv(:,i,e,f) * t_amp ! (in)
                      resid(:,b,i,n) = resid(:,b,i,n) - h2_vovv(:,j,e,f) * t_amp ! (jn)
                      resid(:,e,i,j) = resid(:,e,i,j) - h2_vovv(:,n,b,f) * t_amp ! (be)
                      resid(:,e,n,j) = resid(:,e,n,j) + h2_vovv(:,i,b,f) * t_amp ! (in)(be)
                      resid(:,e,i,n) = resid(:,e,i,n) + h2_vovv(:,j,b,f) * t_amp ! (jn)(be)
                      resid(:,f,i,j) = resid(:,f,i,j) - h2_vovv(:,n,e,b) * t_amp ! (bf)
                      resid(:,f,n,j) = resid(:,f,n,j) + h2_vovv(:,i,e,b) * t_amp ! (in)(bf)
                      resid(:,f,i,n) = resid(:,f,i,n) + h2_vovv(:,j,e,b) * t_amp ! (jn)(bf)
                  end do
                  ! update loop
                  do i = 1,no
                      do j = i+1,no
                          do a = 1,nu
                              do b = a+1,nu
                                  denom = f_oo(i,i) + f_oo(j,j) - f_vv(a,a) - f_vv(b,b)

                                  val = resid(b,a,j,i) - resid(a,b,j,i) - resid(b,a,i,j) + resid(a,b,i,j)
                                  val = val/(denom - shift)

                                  t2(b,a,j,i) =  t2(b,a,j,i) + val
                                  t2(a,b,j,i) = -t2(b,a,j,i)
                                  t2(b,a,i,j) = -t2(b,a,j,i)
                                  t2(a,b,i,j) =  t2(b,a,j,i)

                                  resid(b,a,j,i) =  val
                                  resid(a,b,j,i) = -val
                                  resid(b,a,i,j) = -val
                                  resid(a,b,i,j) =  val
                              end do
                          end do
                      end do
                  end do
                  ! (H(2) * T3)_C terms are vectorized and generally broadcast to diagonal elements, which should
                  ! be 0. Set them to 0 manully (you need to do this).
                  do a = 1, nu
                     resid(a,a,:,:) = 0.0d0
                  end do
                  do i = 1, no
                     resid(:,:,i,i) = 0.0d0
                  end do

              end subroutine update_t2

              subroutine update_t3_p(resid,&
                                     t3_amps,t3_excits,&
                                     t2,&
                                     h1_oo,h1_vv,&
                                     h2_oovv,h2_vvov,h2_vooo,&
                                     h2_oooo,h2_voov,h2_vvvv,&
                                     f_oo,f_vv,&
                                     shift,&
                                     n3,& 
                                     no,nu)

                  integer, intent(in) :: no, nu, n3
                  real(kind=8), intent(in) :: t2(nu,nu,no,no),&
                                              h1_oo(no,no),h1_vv(nu,nu),&
                                              h2_oovv(no,no,nu,nu),&
                                              !h2_vvov(nu,nu,no,nu),&
                                              h2_vvov(nu,nu,nu,no),& ! reordered
                                              !h2_vooo(nu,no,no,no),&
                                              h2_vooo(no,nu,no,no),& ! reordered
                                              h2_oooo(no,no,no,no),&
                                              !h2_voov(nu,no,no,nu),&
                                              h2_voov(no,nu,nu,no),& ! reordered
                                              h2_vvvv(nu,nu,nu,nu),&
                                              f_vv(nu,nu), f_oo(no,no),&
                                              shift

                  integer, intent(inout) :: t3_excits(n3,6)
                  !f2py intent(in,out) :: t3_excits(0:n3-1,0:5)
                  real(kind=8), intent(inout) :: t3_amps(n3)
                  !f2py intent(in,out) :: t3_amps(0:n3-1)

                  real(kind=8), intent(out) :: resid(n3)

                  integer, allocatable :: idx_table(:,:,:,:)
                  integer, allocatable :: loc_arr(:,:)

                  real(kind=8), allocatable :: t3_amps_buff(:), xbuf(:,:,:,:)
                  integer, allocatable :: t3_excits_buff(:,:)

                  !real(kind=8) :: I2_vvov(nu,nu,no,nu)
                  real(kind=8) :: I2_vvov(nu,nu,nu,no) ! reordered
                  !real(kind=8) :: I2_vooo(nu, no, no, no)
                  real(kind=8) :: I2_vooo(no,nu,no,no) ! reordered
                  real(kind=8) :: val, denom, t_amp, res_mm23, hmatel
                  real(kind=8) :: hmatel1, hmatel2, hmatel3, hmatel4
                  integer :: a, b, c, d, i, j, k, l, e, f, m, n, idet, jdet
                  integer :: idx, nloc
                  
                  ! Start the VT3 intermediates at Hbar (factor of 1/2 to compensate for antisymmetrization)
                  I2_vooo = 0.5d0 * h2_vooo 
                  I2_vvov = 0.5d0 * h2_vvov
                  call calc_I2_vooo(I2_vooo,&
                               h2_oovv,&
                               t3_excits,t3_amps,&
                               n3,no,nu)
                  call calc_I2_vvov(I2_vvov,&
                               h2_oovv,&
                               t3_excits,t3_amps,&
                               n3,no,nu)

                  ! Zero the residual container
                  resid = 0.0d0

                  !!!! diagram 1: -A(i/jk) h1a(mi) * t3(abcmjk)
                  !!!! diagram 3: 1/2 A(i/jk) h2(mnij) * t3(abcmnk)
                  ! NOTE: WITHIN THESE LOOPS, H1A(OO) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2  
                  ! allocate new sorting arrays
                  nloc = nu*(nu-1)*(nu-2)/6*no
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nu,nu,nu,no))
                  !!! ABCK LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/-1,nu/), (/3,no/), nu, nu, nu, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,3,6/), nu, nu, nu, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,k)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3_excits(jdet,4); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(oooo) | lmkabc >
                        !hmatel = h2_oooo(l,m,i,j)
                        hmatel = h2_oooo(m,l,j,i)
                        ! compute < ijkabc | h1a(oo) | lmkabc > = -A(ij)A(lm) h1_oo(l,i) * delta(m,j)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (m==j) hmatel1 = -h1_oo(l,i) ! (1)      < ijkabc | h1a(oo) | ljkabc > 
                        if (m==i) hmatel2 = h1_oo(l,j) ! (ij)     < ijkabc | h1a(oo) | likabc > 
                        if (l==j) hmatel3 = h1_oo(m,i) ! (lm)     < ijkabc | h1a(oo) | jmkabc >
                        if (l==i) hmatel4 = -h1_oo(m,j) ! (ij)(lm) < ijkabc | h1a(oo) | imkabc >
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3  + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ik)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3_excits(jdet,4); m = t3_excits(jdet,5);
                           ! compute < ijkabc | h2(oooo) | lmiabc >
                           !hmatel = -h2_oooo(l,m,k,j)
                           hmatel = h2_oooo(m,l,k,j)
                           ! compute < ijkabc | h1a(oo) | lmiabc > = A(jk)A(lm) h1_oo(l,k) * delta(m,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==j) hmatel1 = h1_oo(l,k) ! (1)      < ijkabc | h1a(oo) | ljiabc >
                           if (m==k) hmatel2 = -h1_oo(l,j) ! (jk)     < ijkabc | h1a(oo) | lkiabc >
                           if (l==j) hmatel3 = -h1_oo(m,k) ! (lm)
                           if (l==k) hmatel4 = h1_oo(m,j) ! (jk)(lm)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3_excits(jdet,4); m = t3_excits(jdet,5);
                           ! compute < ijkabc | h2(oooo) | lmjabc >
                           !hmatel = -h2_oooo(l,m,i,k)
                           hmatel = -h2_oooo(m,l,k,i)
                           ! compute < ijkabc | h1a(oo) | lmjabc > = A(ik)A(lm) h1_oo(l,i) * delta(m,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (m==k) hmatel1 = h1_oo(l,i) ! (1)      < ijkabc | h1a(oo) | lkjabc >
                           if (m==i) hmatel2 = -h1_oo(l,k) ! (ik)
                           if (l==k) hmatel3 = -h1_oo(m,i) ! (lm)
                           if (l==i) hmatel4 = h1_oo(m,k) ! (ik)(lm)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCI LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/-1,nu/), (/1,no-2/), nu, nu, nu, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,3,4/), nu, nu, nu, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,i)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        m = t3_excits(jdet,5); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(oooo) | imnabc >
                        !hmatel = h2_oooo(m,n,j,k)
                        hmatel = h2_oooo(n,m,k,j)
                        ! compute < ijkabc | h1a(oo) | imnabc > = -A(jk)A(mn) h1_oo(m,j) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1_oo(m,j)  ! < ijkabc | h1a(oo) | imkabc >
                        if (n==j) hmatel2 = h1_oo(m,k)
                        if (m==k) hmatel3 = h1_oo(n,j)
                        if (m==j) hmatel4 = -h1_oo(n,k)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,j)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3_excits(jdet,5); n = t3_excits(jdet,6);
                           ! compute < ijkabc | h2(oooo) | jmnabc >
                           !hmatel = -h2_oooo(m,n,i,k)
                           hmatel = -h2_oooo(n,m,k,i)
                           ! compute < ijkabc | h1a(oo) | jmnabc > = A(ik)A(mn) h1_oo(m,i) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1_oo(m,i)
                           if (n==i) hmatel2 = -h1_oo(m,k)
                           if (m==k) hmatel3 = -h1_oo(n,i)
                           if (m==i) hmatel4 = h1_oo(n,k)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           m = t3_excits(jdet,5); n = t3_excits(jdet,6);
                           ! compute < ijkabc | h2(oooo) | kmnabc >
                           !hmatel = -h2_oooo(m,n,j,i)
                           hmatel = h2_oooo(n,m,j,i)
                           ! compute < ijkabc | h1a(oo) | kmnabc > = A(ij)A(mn) h1_oo(m,j) * delta(n,i)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==i) hmatel1 = -h1_oo(m,j)
                           if (n==j) hmatel2 = h1_oo(m,i)
                           if (m==i) hmatel3 = h1_oo(n,j)
                           if (m==j) hmatel4 = -h1_oo(n,i)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABCJ LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/-1,nu/), (/2,no-1/), nu, nu, nu, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,3,5/), nu, nu, nu, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_oo,h2_oooo,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,c,j)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        l = t3_excits(jdet,4); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(oooo) | ljnabc >
                        !hmatel = h2_oooo(l,n,i,k)
                        hmatel = h2_oooo(n,l,k,i)
                        ! compute < ijkabc | h1a(oo) | ljnabc > = -A(ik)A(ln) h1_oo(l,i) * delta(n,k)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (n==k) hmatel1 = -h1_oo(l,i)
                        if (n==i) hmatel2 = h1_oo(l,k)
                        if (l==k) hmatel3 = h1_oo(n,i)
                        if (l==i) hmatel4 = -h1_oo(n,k)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ij)
                     idx = idx_table(a,b,c,i)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3_excits(jdet,4); n = t3_excits(jdet,6);
                           ! compute < ijkabc | h2(oooo) | linabc >
                           !hmatel = -h2_oooo(l,n,j,k)
                           hmatel = -h2_oooo(n,l,k,j)
                           ! compute < ijkabc | h1a(oo) | linabc > = A(jk)A(ln) h1_oo(l,j) * delta(n,k)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==k) hmatel1 = h1_oo(l,j)
                           if (n==j) hmatel2 = -h1_oo(l,k)
                           if (l==k) hmatel3 = -h1_oo(n,j)
                           if (l==j) hmatel4 = h1_oo(n,k)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,c,k)
                     if (idx/=0) then
                        do jdet = loc_arr(1,idx), loc_arr(2,idx)
                           l = t3_excits(jdet,4); n = t3_excits(jdet,6);
                           ! compute < ijkabc | h2(oooo) | lknabc >
                           !hmatel = -h2_oooo(l,n,i,j)
                           hmatel = -h2_oooo(n,l,j,i)
                           ! compute < ijkabc | h1a(oo) | lknabc > = A(ij)A(ln) h1_oo(l,i) * delta(n,j)
                           hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                           if (n==j) hmatel1 = h1_oo(l,i)
                           if (n==i) hmatel2 = -h1_oo(l,j)
                           if (l==j) hmatel3 = -h1_oo(n,i)
                           if (l==i) hmatel4 = h1_oo(n,j)
                           hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                           resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                        end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 2: A(a/bc) h1a(ae) * t3(ebcijk)
                  !!!! diagram 4: 1/2 A(c/ab) h2(abef) * t3(ebcijk) 
                  ! NOTE: WITHIN THESE LOOPS, H1A(VV) TERMS ARE DOUBLE-COUNTED SO COMPENSATE BY FACTOR OF 1/2  
                  ! allocate new sorting arrays
                  nloc = no*(no-1)*(no-2)/6*nu
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(no,no,no,nu))
                  !!! IJKA LOOP !!!
                  call get_index_table(idx_table, (/1,no-2/), (/-1,no-1/), (/-1,no/), (/1,nu-2/), no, no, no, nu)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/4,5,6,1/), no, no, no, nu, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_vv,h2_vvvv,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,a)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkaef >
                        !hmatel = h2_vvvv(b,c,e,f)
                        !hmatel = h2_vvvv(e,f,b,c)
                        hmatel = h2_vvvv(f,e,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkaef > = A(bc)A(ef) h1_vv(b,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1_vv(e,b)  !h1_vv(b,e) ! (1)
                        if (b==f) hmatel2 = -h1_vv(e,c) !-h1_vv(c,e) ! (bc)
                        if (c==e) hmatel3 = -h1_vv(f,b) !-h1_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = h1_vv(f,c)  ! h1_vv(c,f) ! (bc)(ef)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkbef >
                        !hmatel = -h2_vvvv(a,c,e,f)
                        !hmatel = -h2_vvvv(e,f,a,c)
                        hmatel = -h2_vvvv(f,e,c,a)
                        ! compute < ijkabc | h1a(vv) | ijkbef > = -A(ac)A(ef) h1_vv(a,e) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1_vv(e,a) !-h1_vv(a,e) ! (1)
                        if (a==f) hmatel2 = h1_vv(e,c)  !h1_vv(c,e) ! (ac)
                        if (c==e) hmatel3 = h1_vv(f,a)  !h1_vv(a,f) ! (ef)
                        if (a==e) hmatel4 = -h1_vv(f,c) !-h1_vv(c,f) ! (ac)(ef)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkcef >
                        !hmatel = -h2_vvvv(b,a,e,f)
                        !hmatel = -h2_vvvv(e,f,b,a)
                        hmatel = h2_vvvv(f,e,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkcef > = -A(ab)A(ef) h1_vv(b,e) * delta(a,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (a==f) hmatel1 = -h1_vv(e,b) !-h1_vv(b,e) ! (1)
                        if (b==f) hmatel2 = h1_vv(e,a)  !h1_vv(a,e) ! (ab)
                        if (a==e) hmatel3 = h1_vv(f,b)  !h1_vv(b,f) ! (ef)
                        if (b==e) hmatel4 = -h1_vv(f,a) !-h1_vv(a,f) ! (ab)(ef)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKB LOOP !!!
                  call get_index_table(idx_table, (/1,no-2/), (/-1,no-1/), (/-1,no/), (/2,nu-1/), no, no, no, nu)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/4,5,6,2/), no, no, no, nu, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_vv,h2_vvvv,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,b)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkdbf >
                        !hmatel = h2_vvvv(a,c,d,f)
                        !hmatel = h2_vvvv(d,f,a,c)
                        hmatel = h2_vvvv(f,d,c,a)
                        ! compute < ijkabc | h1a(vv) | ijkdbf > = A(ac)A(df) h1_vv(a,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = h1_vv(d,a)  !h1_vv(a,d) ! (1)
                        if (a==f) hmatel2 = -h1_vv(d,c) !-h1_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = -h1_vv(f,a) !-h1_vv(a,f) ! (df)
                        if (a==d) hmatel4 = h1_vv(f,c)  !h1_vv(c,f) ! (ac)(df)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ab)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkdaf >
                        !hmatel = -h2_vvvv(b,c,d,f)
                        !hmatel = -h2_vvvv(d,f,b,c)
                        hmatel = -h2_vvvv(f,d,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkdaf > = -A(bc)A(df) h1_vv(b,d) * delta(c,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==f) hmatel1 = -h1_vv(d,b) !-h1_vv(b,d) ! (1)
                        if (b==f) hmatel2 = h1_vv(d,c)  !h1_vv(c,d) ! (bc)
                        if (c==d) hmatel3 = h1_vv(f,b)  !h1_vv(b,f) ! (df)
                        if (b==d) hmatel4 = -h1_vv(f,c) !-h1_vv(c,f) ! (bc)(df)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,c)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); f = t3_excits(jdet,3);
                        ! compute < ijkabc | h2(vvvv) | ijkdcf >
                        !hmatel = -h2_vvvv(a,b,d,f)
                        !hmatel = -h2_vvvv(d,f,a,b)
                        hmatel = -h2_vvvv(f,d,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkdcf > = -A(ab)A(df) h1_vv(a,d) * delta(b,f)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==f) hmatel1 = -h1_vv(d,a) !-h1_vv(a,d) ! (1)
                        if (a==f) hmatel2 = h1_vv(d,b)  !h1_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = h1_vv(f,a)  !h1_vv(a,f) ! (df)
                        if (a==d) hmatel4 = -h1_vv(f,b) !-h1_vv(b,f) ! (ab)(df)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if 
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! IJKC LOOP !!!
                  call get_index_table(idx_table, (/1,no-2/), (/-1,no-1/), (/-1,no/), (/3,nu/), no, no, no, nu)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/4,5,6,3/), no, no, no, nu, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h1_vv,h2_vvvv,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,hmatel1,hmatel2,hmatel3,hmatel4,&
                  !$omp a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(i,j,k,c)
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); e = t3_excits(jdet,2);
                        ! compute < ijkabc | h2(vvvv) | ijkdec >
                        !hmatel = h2_vvvv(a,b,d,e)
                        !hmatel = h2_vvvv(d,e,a,b)
                        hmatel = h2_vvvv(e,d,b,a)
                        ! compute < ijkabc | h1a(vv) | ijkdec > = A(ab)A(de) h1_vv(a,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = h1_vv(d,a)  !h1_vv(a,d) ! (1)
                        if (a==e) hmatel2 = -h1_vv(d,b) !-h1_vv(b,d) ! (ab)
                        if (b==d) hmatel3 = -h1_vv(e,a) !-h1_vv(a,e) ! (de)
                        if (a==d) hmatel4 = h1_vv(e,b)  !h1_vv(b,e) ! (ab)(de)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     ! (ac)
                     idx = idx_table(i,j,k,a)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); e = t3_excits(jdet,2);
                        ! compute < ijkabc | h2(vvvv) | ijkdea >
                        !hmatel = -h2_vvvv(c,b,d,e)
                        !hmatel = -h2_vvvv(d,e,c,b)
                        hmatel = h2_vvvv(e,d,c,b)
                        ! compute < ijkabc | h1a(vv) | ijkdea > = -A(bc)A(de) h1_vv(c,d) * delta(b,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (b==e) hmatel1 = -h1_vv(d,c) !-h1_vv(c,d) ! (1)
                        if (c==e) hmatel2 = h1_vv(d,b)  !h1_vv(b,d) ! (bc)
                        if (b==d) hmatel3 = h1_vv(e,c)  !h1_vv(c,e) ! (de)
                        if (c==d) hmatel4 = -h1_vv(e,b) !-h1_vv(b,e) ! (bc)(de)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(i,j,k,b)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); e = t3_excits(jdet,2);
                        ! compute < ijkabc | h2(vvvv) | ijkdeb >
                        !hmatel = -h2_vvvv(a,c,d,e)
                        !hmatel = -h2_vvvv(d,e,a,c)
                        hmatel = -h2_vvvv(e,d,c,a)
                        ! compute < ijkabc | h1a(vv) | ijkdeb > = -A(ac)A(de) h1_vv(a,d) * delta(c,e)
                        hmatel1 = 0.0d0; hmatel2 = 0.0d0; hmatel3 = 0.0d0; hmatel4 = 0.0d0;
                        if (c==e) hmatel1 = -h1_vv(d,a) !-h1_vv(a,d) ! (1)
                        if (a==e) hmatel2 = h1_vv(d,c)  !h1_vv(c,d) ! (ac)
                        if (c==d) hmatel3 = h1_vv(e,a)  !h1_vv(a,e) ! (de)
                        if (a==d) hmatel4 = -h1_vv(e,c) !-h1_vv(c,e) ! (ac)(de)
                        hmatel = hmatel + 0.5d0*(hmatel1 + hmatel2 + hmatel3 + hmatel4)
                        resid(idet) = resid(idet) + hmatel*t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !!!! diagram 5: A(i/jk)A(a/bc) h2(amie) * t3(ebcmjk)
                  ! allocate sorting arrays (can be reused for each permutation)
                  nloc = (nu-1)*(nu-2)/2*(no-1)*(no-2)/2
                  allocate(loc_arr(2,nloc))
                  allocate(idx_table(nu,nu,no,no))
                  !!! ABIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/1,no-2/), (/-1,no-1/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,4,5/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnabf >
                        !hmatel = h2_voov(c,n,k,f)
                        hmatel = h2_voov(n,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnbcf >
                        !hmatel = h2_voov(a,n,k,f)
                        hmatel = h2_voov(n,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnacf >
                        !hmatel = -h2_voov(b,n,k,f)
                        hmatel = -h2_voov(n,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknabf >
                        !hmatel = h2_voov(c,n,i,f)
                        hmatel = h2_voov(n,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknbcf >
                        !hmatel = h2_voov(a,n,i,f)
                        hmatel = h2_voov(n,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknacf >
                        !hmatel = -h2_voov(b,n,i,f)
                        hmatel = -h2_voov(n,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknabf >
                        !hmatel = -h2_voov(c,n,j,f)
                        hmatel = -h2_voov(n,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknbcf >
                        !hmatel = -h2_voov(a,n,j,f)
                        hmatel = -h2_voov(n,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknacf >
                        !hmatel = h2_voov(b,n,j,f)
                        hmatel = h2_voov(n,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIJ LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-2,nu/), (/1,no-2/), (/-1,no-1/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,3,4,5/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnaec >
                        !hmatel = h2_voov(b,n,k,e)
                        hmatel = h2_voov(n,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnbec >
                        !hmatel = -h2_voov(a,n,k,e)
                        hmatel = -h2_voov(n,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijnaeb >
                        !hmatel = -h2_voov(c,n,k,e)
                        hmatel = -h2_voov(n,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknaec >
                        !hmatel = h2_voov(b,n,i,e)
                        hmatel = h2_voov(n,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknbec >
                        !hmatel = -h2_voov(a,n,i,e)
                        hmatel = -h2_voov(n,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jknaeb >
                        !hmatel = -h2_voov(c,n,i,e)
                        hmatel = -h2_voov(n,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknaec >
                        !hmatel = -h2_voov(b,n,j,e)
                        hmatel = -h2_voov(n,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknbec >
                        !hmatel = h2_voov(a,n,j,e)
                        hmatel = h2_voov(n,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | iknaeb >
                        !hmatel = h2_voov(c,n,j,e)
                        hmatel = h2_voov(n,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIJ LOOP !!!
                  call get_index_table(idx_table, (/2,nu-1/), (/-1,nu/), (/1,no-2/), (/-1,no-1/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/2,3,4,5/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijndbc >
                        !hmatel = h2_voov(a,n,k,d)
                        hmatel = h2_voov(n,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijndac >
                        !hmatel = -h2_voov(b,n,k,d)
                        hmatel = -h2_voov(n,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ijndab >
                        !hmatel = h2_voov(c,n,k,d)
                        hmatel = h2_voov(n,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jkndbc >
                        !hmatel = h2_voov(a,n,i,d)
                        hmatel = h2_voov(n,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jkndac >
                        !hmatel = -h2_voov(b,n,i,d)
                        hmatel = -h2_voov(n,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | jkndab >
                        !hmatel = h2_voov(c,n,i,d)
                        hmatel = h2_voov(n,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ikndbc >
                        !hmatel = -h2_voov(a,n,j,d)
                        hmatel = -h2_voov(n,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ikndac >
                        !hmatel = h2_voov(b,n,j,d)
                        hmatel = h2_voov(n,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); n = t3_excits(jdet,6);
                        ! compute < ijkabc | h2(voov) | ikndab >
                        !hmatel = -h2_voov(c,n,j,d)
                        hmatel = -h2_voov(n,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABIK LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/1,no-2/), (/-2,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,4,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkabf >
                        !hmatel = h2_voov(c,m,j,f)
                        hmatel = h2_voov(m,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkbcf >
                        !hmatel = h2_voov(a,m,j,f)
                        hmatel = h2_voov(m,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkacf >
                        !hmatel = -h2_voov(b,m,j,f)
                        hmatel = -h2_voov(m,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkabf >
                        !hmatel = -h2_voov(c,m,i,f)
                        hmatel = -h2_voov(m,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkbcf >
                        !hmatel = -h2_voov(a,m,i,f)
                        hmatel = -h2_voov(m,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkacf >
                        !hmatel = h2_voov(b,m,i,f)
                        hmatel = h2_voov(m,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjabf >
                        !hmatel = -h2_voov(c,m,k,f)
                        hmatel = -h2_voov(m,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjbcf >
                        !hmatel = -h2_voov(a,m,k,f)
                        hmatel = -h2_voov(m,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjacf >
                        !hmatel = h2_voov(b,m,k,f)
                        hmatel = h2_voov(m,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACIK LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-2,nu/), (/1,no-2/), (/-2,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,3,4,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkaec >
                        !hmatel = h2_voov(b,m,j,e)
                        hmatel = h2_voov(m,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkbec >
                        !hmatel = -h2_voov(a,m,j,e)
                        hmatel = -h2_voov(m,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkaeb >
                        !hmatel = -h2_voov(c,m,j,e)
                        hmatel = -h2_voov(m,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkaec >
                        !hmatel = -h2_voov(b,m,i,e)
                        hmatel = -h2_voov(m,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkbec >
                        !hmatel = h2_voov(a,m,i,e)
                        hmatel = h2_voov(m,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkaeb >
                        !hmatel = h2_voov(c,m,i,e)
                        hmatel = h2_voov(m,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjaec >
                        !hmatel = -h2_voov(b,m,k,e)
                        hmatel = -h2_voov(m,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjbec >
                        !hmatel = h2_voov(a,m,k,e)
                        hmatel = h2_voov(m,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjaeb >
                        !hmatel = h2_voov(c,m,k,e)
                        hmatel = h2_voov(m,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCIK LOOP !!!
                  call get_index_table(idx_table, (/2,nu-1/), (/-1,nu/), (/1,no-2/), (/-2,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/2,3,4,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkdbc >
                        !hmatel = h2_voov(a,m,j,d)
                        hmatel = h2_voov(m,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkdac >
                        !hmatel = -h2_voov(b,m,j,d)
                        hmatel = -h2_voov(m,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imkdab >
                        !hmatel = h2_voov(c,m,j,d)
                        hmatel = h2_voov(m,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkdbc >
                        !hmatel = -h2_voov(a,m,i,d)
                        hmatel = -h2_voov(m,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkdac >
                        !hmatel = h2_voov(b,m,i,d)
                        hmatel = h2_voov(m,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | jmkdab >
                        !hmatel = -h2_voov(c,m,i,d)
                        hmatel = -h2_voov(m,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (jk)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjdbc >
                        !hmatel = -h2_voov(a,m,k,d)
                        hmatel = -h2_voov(m,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(jk)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjdac >
                        !hmatel = h2_voov(b,m,k,d)
                        hmatel = h2_voov(m,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(jk)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); m = t3_excits(jdet,5);
                        ! compute < ijkabc | h2(voov) | imjdab >
                        !hmatel = -h2_voov(c,m,k,d)
                        hmatel = -h2_voov(m,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ABJK LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-1,nu-1/), (/2,no-1/), (/-1,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,2,5,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkabf >
                        !hmatel = h2_voov(c,l,i,f)
                        hmatel = h2_voov(l,f,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkbcf >
                        !hmatel = h2_voov(a,l,i,f)
                        hmatel = h2_voov(l,f,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkacf >
                        !hmatel = -h2_voov(b,l,i,f)
                        hmatel = -h2_voov(l,f,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likabf >
                        !hmatel = -h2_voov(c,l,j,f)
                        hmatel = -h2_voov(l,f,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likbcf >
                        !hmatel = -h2_voov(a,l,j,f)
                        hmatel = -h2_voov(l,f,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likacf >
                        !hmatel = h2_voov(b,l,j,f)
                        hmatel = h2_voov(l,f,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijabf >
                        !hmatel = h2_voov(c,l,k,f)
                        hmatel = h2_voov(l,f,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijbcf >
                        !hmatel = h2_voov(a,l,k,f)
                        hmatel = h2_voov(l,f,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        f = t3_excits(jdet,3); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijacf >
                        !hmatel = -h2_voov(b,l,k,f)
                        hmatel = -h2_voov(l,f,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! ACJK LOOP !!!
                  call get_index_table(idx_table, (/1,nu-2/), (/-2,nu/), (/2,no-1/), (/-1,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/1,3,5,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkaec >
                        !hmatel = h2_voov(b,l,i,e)
                        hmatel = h2_voov(l,e,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkbec >
                        !hmatel = -h2_voov(a,l,i,e)
                        hmatel = -h2_voov(l,e,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkaeb >
                        !hmatel = -h2_voov(c,l,i,e)
                        hmatel = -h2_voov(l,e,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likaec >
                        !hmatel = -h2_voov(b,l,j,e)
                        hmatel = -h2_voov(l,e,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likbec >
                        !hmatel = h2_voov(a,l,j,e)
                        hmatel = h2_voov(l,e,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likaeb >
                        !hmatel = h2_voov(c,l,j,e)
                        hmatel = h2_voov(l,e,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijaec >
                        !hmatel = h2_voov(b,l,k,e)
                        hmatel = h2_voov(l,e,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijbec >
                        !hmatel = -h2_voov(a,l,k,e)
                        hmatel = -h2_voov(l,e,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (bc)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        e = t3_excits(jdet,2); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijaeb >
                        !hmatel = -h2_voov(c,l,k,e)
                        hmatel = -h2_voov(l,e,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  !!! BCJK LOOP !!!
                  call get_index_table(idx_table, (/2,nu-1/), (/-1,nu/), (/2,no-1/), (/-1,no/), nu, nu, no, no)
                  call sort4(t3_excits, t3_amps, loc_arr, idx_table, (/2,3,5,6/), nu, nu, no, no, nloc, n3, resid)
                  !!!! BEGIN OMP PARALLEL SECTION !!!!
                  !$omp parallel shared(resid,&
                  !$omp t3_excits,&
                  !$omp t3_amps,&
                  !$omp loc_arr,idx_table,&
                  !$omp h2_voov,&
                  !$omp no,nu,n3),&
                  !$omp private(hmatel,a,b,c,d,i,j,k,l,e,f,m,n,idet,jdet,&
                  !$omp idx)
                  !$omp do schedule(static)
                  do idet = 1, n3
                     a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                     i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                     ! (1)
                     idx = idx_table(b,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkdbc >
                        !hmatel = h2_voov(a,l,i,d)
                        hmatel = h2_voov(l,d,a,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)
                     idx = idx_table(a,c,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkdac >
                        !hmatel = -h2_voov(b,l,i,d)
                        hmatel = -h2_voov(l,d,b,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)
                     idx = idx_table(a,b,j,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | ljkdab >
                        !hmatel = h2_voov(c,l,i,d)
                        hmatel = h2_voov(l,d,c,i)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ij)
                     idx = idx_table(b,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likdbc >
                        !hmatel = -h2_voov(a,l,j,d)
                        hmatel = -h2_voov(l,d,a,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ij)
                     idx = idx_table(a,c,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likdac >
                        !hmatel = h2_voov(b,l,j,d)
                        hmatel = h2_voov(l,d,b,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ij)
                     idx = idx_table(a,b,i,k)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | likdab >
                        !hmatel = -h2_voov(c,l,j,d)
                        hmatel = -h2_voov(l,d,c,j)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ik)
                     idx = idx_table(b,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijdbc >
                        !hmatel = h2_voov(a,l,k,d)
                        hmatel = h2_voov(l,d,a,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ab)(ik)
                     idx = idx_table(a,c,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijdac >
                        !hmatel = -h2_voov(b,l,k,d)
                        hmatel = -h2_voov(l,d,b,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                     ! (ac)(ik)
                     idx = idx_table(a,b,i,j)
                     if (idx/=0) then
                     do jdet = loc_arr(1,idx), loc_arr(2,idx)
                        d = t3_excits(jdet,1); l = t3_excits(jdet,4);
                        ! compute < ijkabc | h2(voov) | lijdab >
                        !hmatel = h2_voov(c,l,k,d)
                        hmatel = h2_voov(l,d,c,k)
                        resid(idet) = resid(idet) + hmatel * t3_amps(jdet)
                     end do
                     end if
                  end do
                  !$omp end do
                  !$omp end parallel
                  !!!! END OMP PARALLEL SECTION !!!!
                  ! deallocate sorting arrays
                  deallocate(loc_arr,idx_table)

                  !
                  ! Moment contributions
                  !
                  allocate(xbuf(no,no,nu,nu))
                  do a = 1,nu
                     do b = 1,nu
                        do i = 1,no
                           do j = 1,no
                              xbuf(j,i,b,a) = t2(b,a,j,i)
                           end do
                        end do
                     end do
                  end do
                  !$omp parallel shared(resid,t3_excits,xbuf,I2_vooo,n3),&
                  !$omp private(idet,a,b,c,i,j,k,m)
                  !$omp do schedule(static)
                  do idet = 1, n3
                      a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                      do m = 1, no
                          ! -A(k/ij)A(a/bc) h2(amij) * t2(bcmk)
                          resid(idet) = resid(idet) - I2_vooo(m,a,i,j) * xbuf(m,k,b,c)
                          resid(idet) = resid(idet) + I2_vooo(m,b,i,j) * xbuf(m,k,a,c)
                          resid(idet) = resid(idet) + I2_vooo(m,c,i,j) * xbuf(m,k,b,a)
                          resid(idet) = resid(idet) + I2_vooo(m,a,k,j) * xbuf(m,i,b,c)
                          resid(idet) = resid(idet) - I2_vooo(m,b,k,j) * xbuf(m,i,a,c)
                          resid(idet) = resid(idet) - I2_vooo(m,c,k,j) * xbuf(m,i,b,a)
                          resid(idet) = resid(idet) + I2_vooo(m,a,i,k) * xbuf(m,j,b,c)
                          resid(idet) = resid(idet) - I2_vooo(m,b,i,k) * xbuf(m,j,a,c)
                          resid(idet) = resid(idet) - I2_vooo(m,c,i,k) * xbuf(m,j,b,a)
                      end do
                  end do
                  !$omp end do
                  !$omp end parallel
                  deallocate(xbuf)

                  !$omp parallel shared(resid,t3_excits,t2,I2_vvov,n3),&
                  !$omp private(idet,a,b,c,i,j,k,e)
                  !$omp do schedule(static)
                  do idet = 1, n3
                      a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                      do e = 1, nu
                           ! A(i/jk)(c/ab) h2(abie) * t2(ecjk)
                          resid(idet) = resid(idet) + I2_vvov(e,a,b,i) * t2(e,c,j,k)
                          resid(idet) = resid(idet) - I2_vvov(e,c,b,i) * t2(e,a,j,k)
                          resid(idet) = resid(idet) - I2_vvov(e,a,c,i) * t2(e,b,j,k)
                          resid(idet) = resid(idet) - I2_vvov(e,a,b,j) * t2(e,c,i,k)
                          resid(idet) = resid(idet) + I2_vvov(e,c,b,j) * t2(e,a,i,k)
                          resid(idet) = resid(idet) + I2_vvov(e,a,c,j) * t2(e,b,i,k)
                          resid(idet) = resid(idet) - I2_vvov(e,a,b,k) * t2(e,c,j,i)
                          resid(idet) = resid(idet) + I2_vvov(e,c,b,k) * t2(e,a,j,i)
                          resid(idet) = resid(idet) + I2_vvov(e,a,c,k) * t2(e,b,j,i)
                      end do
                  end do
                  !$omp end do
                  !$omp end parallel

                  ! Update t3 vector
                  !$omp parallel shared(resid,t3_excits,t3_amps,f_oo,f_vv,n3,shift),&
                  !$omp private(idet,a,b,c,i,j,k,denom)
                  !$omp do schedule(static)
                  do idet = 1,n3
                      a = t3_excits(idet,1); b = t3_excits(idet,2); c = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); k = t3_excits(idet,6);
                      denom = f_oo(i,i) + f_oo(j,j) + f_oo(k,k) - f_vv(a,a) - f_vv(b,b) - f_vv(c,c)
                      resid(idet) = resid(idet)/(denom - shift)
                      t3_amps(idet) = t3_amps(idet) + resid(idet)
                  end do
                  !$omp end do
                  !$omp end parallel

              end subroutine update_t3_p

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!! INTERMEDIATES FUNCTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine calc_I2_vooo(I2_vooo,&
                              h2_oovv,&
                              t3_excits,t3_amps,&
                              n3,no,nu)

                  integer, intent(in) :: no, nu, n3
                  integer, intent(in) :: t3_excits(n3,6)
                  real(kind=8), intent(in) :: t3_amps(n3)
                  real(kind=8), intent(in) :: h2_oovv(no,no,nu,nu)
                  real(kind=8), intent(inout) :: I2_vooo(no,nu,no,no)

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp 

                  do idet = 1, n3
                      t_amp = t3_amps(idet)
                      ! I2(amij) <- A(ij) [A(n/ij)A(a/ef) h2(mnef) * t3(aefijn)]
                      a = t3_excits(idet,1); e = t3_excits(idet,2); f = t3_excits(idet,3);
                      i = t3_excits(idet,4); j = t3_excits(idet,5); n = t3_excits(idet,6);
                      I2_vooo(:,a,i,j) = I2_vooo(:,a,i,j) + h2_oovv(:,n,e,f) * t_amp ! (1)
                      I2_vooo(:,a,n,j) = I2_vooo(:,a,n,j) - h2_oovv(:,i,e,f) * t_amp ! (in)
                      I2_vooo(:,a,i,n) = I2_vooo(:,a,i,n) - h2_oovv(:,j,e,f) * t_amp ! (jn)
                      I2_vooo(:,e,i,j) = I2_vooo(:,e,i,j) - h2_oovv(:,n,a,f) * t_amp ! (ae)
                      I2_vooo(:,e,n,j) = I2_vooo(:,e,n,j) + h2_oovv(:,i,a,f) * t_amp ! (in)(ae)
                      I2_vooo(:,e,i,n) = I2_vooo(:,e,i,n) + h2_oovv(:,j,a,f) * t_amp ! (jn)(ae)
                      I2_vooo(:,f,i,j) = I2_vooo(:,f,i,j) - h2_oovv(:,n,e,a) * t_amp ! (af)
                      I2_vooo(:,f,n,j) = I2_vooo(:,f,n,j) + h2_oovv(:,i,e,a) * t_amp ! (in)(af)
                      I2_vooo(:,f,i,n) = I2_vooo(:,f,i,n) + h2_oovv(:,j,e,a) * t_amp ! (jn)(af)
                  end do
                  ! antisymmetrize
                  do i = 1,no
                     do j = i+1,no
                        do a = 1,nu
                           do m = 1,no
                              I2_vooo(m,a,i,j) = I2_vooo(m,a,i,j) - I2_vooo(m,a,j,i)
                              I2_vooo(m,a,j,i) = -I2_vooo(m,a,i,j)
                           end do
                        end do
                     end do
                  end do
      end subroutine calc_I2_vooo

      subroutine calc_I2_vvov(I2_vvov,&
                              h2_oovv,&
                              t3_excits,t3_amps,&
                              n3,no,nu)

                  integer, intent(in) :: no, nu, n3
                  integer, intent(in) :: t3_excits(n3,6)
                  real(kind=8), intent(in) :: t3_amps(n3)
                  real(kind=8), intent(in) :: h2_oovv(no,no,nu,nu)

                  real(kind=8), intent(inout) :: I2_vvov(nu,nu,nu,no) ! reordered

                  integer :: idet, a, b, c, i, j, k, m, n, e, f
                  real(kind=8) :: t_amp 
                  real(kind=8), allocatable :: intbuf(:,:,:,:)

                  allocate(intbuf(nu,nu,no,no))
                  do i = 1,no
                     do j = 1,no
                        do a = 1,nu
                           do b = 1,nu
                              intbuf(b,a,j,i) = h2_oovv(j,i,b,a)
                           end do
                        end do
                     end do
                  end do
                  do idet = 1, n3
                      t_amp = t3_amps(idet)
                      ! I2(abie) <- A(ab) [A(i/mn)A(f/ab) -h2(mnef) * t3(abfimn)]
                      a = t3_excits(idet,1); b = t3_excits(idet,2); f = t3_excits(idet,3);
                      i = t3_excits(idet,4); m = t3_excits(idet,5); n = t3_excits(idet,6);
                      I2_vvov(:,a,b,i) = I2_vvov(:,a,b,i) - intbuf(:,f,m,n) * t_amp ! (1)
                      I2_vvov(:,a,b,m) = I2_vvov(:,a,b,m) + intbuf(:,f,i,n) * t_amp ! (im)
                      I2_vvov(:,a,b,n) = I2_vvov(:,a,b,n) + intbuf(:,f,m,i) * t_amp ! (in)
                      I2_vvov(:,f,b,i) = I2_vvov(:,f,b,i) + intbuf(:,a,m,n) * t_amp ! (af)
                      I2_vvov(:,f,b,m) = I2_vvov(:,f,b,m) - intbuf(:,a,i,n) * t_amp ! (im)(af)
                      I2_vvov(:,f,b,n) = I2_vvov(:,f,b,n) - intbuf(:,a,m,i) * t_amp ! (in)(af)
                      I2_vvov(:,a,f,i) = I2_vvov(:,a,f,i) + intbuf(:,b,m,n) * t_amp ! (bf)
                      I2_vvov(:,a,f,m) = I2_vvov(:,a,f,m) - intbuf(:,b,i,n) * t_amp ! (im)(bf)
                      I2_vvov(:,a,f,n) = I2_vvov(:,a,f,n) - intbuf(:,b,m,i) * t_amp ! (in)(bf)
                  end do
                  deallocate(intbuf)
                  ! antisymmetrize
                  do i = 1,no
                     do a = 1,nu
                        do b = a+1,nu
                           do e = 1,nu
                              I2_vvov(e,a,b,i) = I2_vvov(e,a,b,i) - I2_vvov(e,b,a,i)
                              I2_vvov(e,b,a,i) = -I2_vvov(e,a,b,i)
                           end do
                        end do
                     end do
                  end do
      end subroutine calc_I2_vvov

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
              if (rng1(1) < 0 .and. rng2(1) < 0 .and. rng3(1) < 0 .and. rng4(1) < 0) then ! p < q < r < s
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

end module ccsdt_p_loops
