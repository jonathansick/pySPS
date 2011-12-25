! fsps.f90 -- a module intended to be wrapped by f2py as a python module
! for making stellar pops

module fsps
    use sps_vars; use nrtype; use sps_utils
    implicit none

    !type(params) :: pset
    !type(COMPSPOUT), dimension(ntfull) :: ocompsp
    !integer :: zi

    !integer, intent(in) :: _imf_type, _dust_type, _zmet, _sfh,
    
    !real, intent(in) :: _imf1, imf2, _imf3, _vdmc, _mdave,
    !real, intent(in) ::_dell, _delt, _fbhb, _pagb
    !real, intent(in) :: _tau, _const, _tage, _fburst, _tburst,
    !real, intent(in) :: _dust_tesc, _dust1, _dust2
    !real :: _dust_clumps, _frac_no_dust, _dust_index, _mwr
    !real :: _wgp1, _wgp2, _wgp3, _duste_gamma, _duste_qpah
    !character(len=256), intent(in) :: model_name,
    !character(len=256) :: output_path

contains
    ! General setup that loads isochrones and stellar libraries
    subroutine setup() ! _vega, _redshift_colors
        !integer, intent(in) :: _vega, _redshift_colors
        !compute_vega_mags = _vega
        !redshift_colors = _redshift_colors
        !! Load isochrones and stellar libraries
        !call sps_setup(-1)
        write (*,*) 'Hello, brave world!'
    end subroutine

    !subroutine setup_all_ssp(_imf_type, _imf1, _imf2, _imf3, _vdmc, &
    !    _mdave, _dell, _delt, _sbss, _fbhb, _pagb)
    !    ! Dependent on imf, sbss, fbhb, delt, dell, pagb, redgb
    !    imf_type = _imf_type
    !    pset%imf1 = _imf1
    !    pset%imf2 = _imf2
    !    pset%imf3 = _imf3
    !    pset%vdmc = _vdmc
    !    pset%mdave = _mdave
    !    pset%dell = _dell
    !    pset%delt = _delt
    !    pset%sbss = _sbss
    !    pset%fbhb = _fbhb
    !    pset%pagb = _pagb
    !    
    !    !set up all SSPs at once (i.e., all metallicities)
    !    spec_ssp_zz = 0.0
    !    do zi=1,nz
    !        ! need to set blue HB and delta AGB before this
    !        pset%zmet = zi
    !        call ssp_gen(pset,mass_ssp_zz(zi,:), &
    !            lbol_ssp_zz(zi,:), spec_ssp_zz(zi,:,:))
    !    end do
    !end subroutine

    !subroutine comp_sp(model_name, _dust_type, _zmet, _sfh, _tau, _const, &
    !        _fburst, _tburst, _dust_tesc, _dust1, _dust2, _dust_clumps, &
    !        _frac_no_dust, _dust_index, _mwr, _wgp1, _wgp2, _wgp3, &
    !        _duste_gamma, _duste_umin, _duste_qpah, _tage)

    !    dust_type = _dust_type
    !    pset%zmet = _zmet
    !    pset%sfh = _sfh
    !    pset%tau = _tau
    !    pset%const = _const
    !    pset%tage = _tage
    !    pset%fburst = _fburst
    !    pset%tburst = _tburst
    !    pset%dust_tesc = _dust_tesc
    !    pset%dust1 = _dust1
    !    pset%dust2 = _dust2
    !    pset%dust_clumps = _dust_clumps
    !    pset%frac_no_dust = _frac_no_dust
    !    pset%dust_index = _dust_index
    !    pset%mwr = _mwr
    !    pset%wgp1 = _wgp1
    !    pset%wgp2 = _wgp2
    !    pset%wpg3 = _wgp3
    !    pset%duste_gamma = _duste_gamma
    !    pset%duste_umin = _duste_qpah
    !    ! For calling COMPSP
    !    ! depends on zmet, SFH, dust
    !    ! we want to directly return the results, for now use file output
    !    output_path = trim(model_name)//'.out'
    !    call compsp(3,1,output_path,mass_ssp,lbol_ssp,spec_ssp,pset,ocompsp)
    !end subroutine

end module
