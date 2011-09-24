! fspsq.f90 -- allows FSPS to generate modules given an input files with
! model parameters. Booting FSPS once and running many models saves on
! disk I/O of isochrones when generating a large composite stellar pop
! libray

program fspsq
      implicit none

      character(len=32) :: arg
      character(len=64) :: input_path
      integer :: i, len
      real dummy
      !do i = 1, command_argument_count()
      !  call get_command_argument(i, arg)
      !  print '(a)', arg
      !end do
      call get_command_argument(1, input_path)
      write (*,*) trim(input_path)
      open(15,file=trim(input_path),status='OLD')
      len = 0 ! number of lines
      do
        read(15,end=20,err=30) dummy
      end do

20    rewind(15)
      allocate(x(2,len))
      do i=1,len
        read(15,*,err=30) x(1,i),x(2,i)
      end do
      close(15)
      write (*,*) len,' data elements read'
      stop

30    write (*,*) 'I/O error occurred'

end program fspsq
