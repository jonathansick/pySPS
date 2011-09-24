! fspsq.f90 -- allows FSPS to generate modules given an input files with
! model parameters. Booting FSPS once and running many models saves on
! disk I/O of isochrones when generating a large composite stellar pop
! libray

program fspsq
      implicit none

      character(len=32) :: arg
      character(len=64) :: input_path
      integer :: i, col1, col2
      character(len=6) :: model_name
      !do i = 1, command_argument_count()
      !  call get_command_argument(i, arg)
      !  print '(a)', arg
      !end do
      call get_command_argument(1, input_path)
      write (*,*) trim(input_path)
      open(15,file=trim(input_path),status='OLD')
      	read(15,*) model_name, col1, col2
		write (*,*) model_name, col1, col2
		col1 = col1*2
		col2 = col2+1
		write (*,*) col1, col2
      close(15)
      ! write (*,*) len,' data elements read'
      stop

30    write (*,*) 'I/O error occurred'

end program fspsq
