! fspsq.f90 -- allows FSPS to generate modules given an input files with
! model parameters. Booting FSPS once and running many models saves on
! disk I/O of isochrones when generating a large composite stellar pop
! libray

program fspsq
    ! setup; note this pollutes our global namespace
    use sps_vars; use nrtype; use sps_utils
    implicit none

    character(len=64) :: input_path
    integer :: imodel
    character(len=256) :: model_name, output_path
    ! character(len=4) :: isoc_type
    ! integer :: compute_vega_mags, dust_type, imf_type, redshift_colors
    ! integer :: time_res_incr, zmet, sfh
    ! real :: zred, const, tau, sf_start, tage, fburst, tburst, imf1, imf2
    ! real :: imf3, vdmc, mdave, dust_tesc, dust1, dust2, dust_clumps
    ! real :: frac_nodust, dust_index, mwr, uvb
    ! integer :: wgp1, wgp2, wgp3
    ! real :: dell, delt, sbss, fbhb, pagb
    
    ! These are parameters that shouldn't be read:
    integer :: fake_time_res_incr
    character(len=4) :: fake_isoc_type
    
    !define variable for SSP spectrum
    REAL(SP), DIMENSION(ntfull,nspec)  :: spec_ssp
    !define variables for Mass and Lbol info
    REAL(SP), DIMENSION(ntfull)    :: mass_ssp,lbol_ssp
    ! CHARACTER(100) :: file1='', file2=''
    !structure containing all necessary parameters
    TYPE(PARAMS) :: pset
    !define structure for CSP spectrum
    TYPE(COMPSPOUT), DIMENSION(ntfull) :: ocompsp
    REAL(SP) :: ssfr6,ssfr7,ssfr8,ave_age
    
    ! Lets compute an SSP, solar metallicity, with a Chabrier IMF
    ! with no dust, and the 'default' assumptions regarding the 
    ! locations of the isochrones

    imf_type  = 1             !define the IMF (1=Chabrier 2003)
                              !see sps_vars.f90 for details of this var
    pset%zmet = 20            !define the metallicity (see the manual)
                              !20 = solar metallacity
    
    ! read in the isochrones and spectral libraries
    CALL SPS_SETUP(pset%zmet)

    call get_command_argument(1, input_path)
    write (*,*) trim(input_path)
    imodel = 0
    open(15,file=trim(input_path),status='OLD')
        read(15,*) model_name, compute_vega_mags, dust_type, imf_type, &
            fake_isoc_type, redshift_colors, fake_time_res_incr, pset%zred, &
            pset%zmet, &
            pset%sfh, pset%tau, pset%const, pset%sf_start, pset%tage, &
            pset%fburst, pset%tburst, pset%imf1, pset%imf2, pset%imf3, &
            pset%vdmc, pset%mdave, pset%dust_tesc, pset%dust1, pset%dust2, &
            pset%dust_clumps, pset%frac_nodust, pset%dust_index, pset%mwr, &
            pset%uvb, pset%wgp1, pset%wgp2, pset%wgp3, pset%dell,pset%delt, &
            pset%sbss, pset%fbhb, pset%pagb
        write (*,*) trim(model_name), pset%tau
        !compute the SSP
        CALL SSP_GEN(pset,mass_ssp,lbol_ssp,spec_ssp)
        !compute mags and write out mags and spec for SSP
        output_path = trim(model_name)//'.out'
        CALL COMPSP(3,1,output_path,mass_ssp,lbol_ssp,spec_ssp,pset,ocompsp)
        imodel = imodel + 1
    close(15)
    write (*,*) imodel, ' model(s) processed'
    stop

30  write (*,*) 'I/O error occurred'

end program fspsq
