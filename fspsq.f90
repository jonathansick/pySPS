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
    character(len=1) :: s_sfh, s_dust_type, s_imf_type, s_vega, s_redshift
    character(len=2) :: s_zmet
    
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
    
    ! Gather common settings and model spec file from command line
    call get_command_argument(1, input_path)
    call get_command_argument(2, s_sfh)
    call get_command_argument(3, s_zmet)
    call get_command_argument(4, s_dust_type) ! sps_vars
    call get_command_argument(5, s_imf_type) ! sps_vars
    call get_command_argument(6, s_vega) ! sps_vars
    call get_command_argument(7, s_redshift) ! sps_vars
    ! Cast command line parameters to integers
    read (s_sfh,*) pset%sfh
    read (s_zmet,*) pset%zmet
    read (s_dust_type,*) dust_type
    read (s_imf_type,*) imf_type
    read (s_vega,*) compute_vega_mags
    read (s_redshift,*) redshift_colors
    
    write (*,*) trim(input_path)
    imodel = 0
    open(15,file=trim(input_path),status='OLD')
        ! read the line generated by fspsq.py ParameterSet.command()
        read(15,*) model_name, pset%zred, pset%tau, pset%const, &
            pset%sf_start, pset%tage, &
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
