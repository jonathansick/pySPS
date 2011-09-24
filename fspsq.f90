! fspsq.f90 -- allows FSPS to generate modules given an input files with
! model parameters. Booting FSPS once and running many models saves on
! disk I/O of isochrones when generating a large composite stellar pop
! libray

program fspsq
      implicit none

      character(len=64) :: input_path
      integer :: imodel
      character(len=256) :: model_name
      character(len=4) :: isoc_type
      integer :: compute_vega_mags, dust_type, imf_type, redshift_colors
      integer :: time_res_incr, zmet, sfh
      real :: zred, const, tau, sf_start, tage, fburst, tburst, imf1, imf2
      real :: imf3, vdmc, mdave, dust_tesc, dust1, dust2, dust_clumps
      real :: frac_nodust, dust_index, mwr, uvb
      integer :: wgp1, wgp2, wgp3
      real :: dell, delt, sbss, fbhb, pagb
      
      call get_command_argument(1, input_path)
      write (*,*) trim(input_path)
      imodel = 0
      open(15,file=trim(input_path),status='OLD')
      	read(15,*) model_name, compute_vega_mags, dust_type, imf_type, &
			isoc_type, redshift_colors, time_res_incr, zred, zmet, sfh, &
			tau, const, sf_start, tage, fburst, tburst, imf1, imf2, imf3, &
			vdmc, mdave, dust_tesc, dust1, dust2, dust_clumps, frac_nodust, &
			dust_index, mwr, uvb, wgp1, wgp2, wgp3, &
			dell, delt, sbss, fbhb, pagb
		write (*,*) trim(model_name), tau
		imodel = imodel + 1
      close(15)
      write (*,*) imodel, ' model(s) processed'
      stop

30    write (*,*) 'I/O error occurred'

end program fspsq
