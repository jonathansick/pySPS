! fsps.f90 -- a module intended to be wrapped by f2py as a python module
! for making stellar pops

module fsps
    use sps_vars; use nrtype; use sps_utils
    implicit none

    type(params) :: pset
    !f2py intent(hide) pset
    type(COMPSPOUT), dimension(ntfull) :: ocompsp
    !f2py intent(hide) ocompsp
    integer :: zi, my_foo_var

contains
    ! General setup that loads isochrones and stellar libraries
    subroutine setup(use_vega_mags, apply_redshift)
        integer, intent(in) :: use_vega_mags, apply_redshift
        write (*,*) 'starting setup()'
        compute_vega_mags = use_vega_mags
        redshift_colors = apply_redshift
        ! Load isochrones and stellar libraries
        call sps_setup(-1)
        write (*,*) 'setup() complete'
    end subroutine

    subroutine setup_all_ssp(imf, imf1, imf2, imf3, vdmc, &
            mdave, dell, delt, sbss, fbhb, pagb)
        integer, intent(in) :: imf
        real, intent(in) :: imf1, imf2, imf3, vdmc, mdave
        real, intent(in) :: dell, delt, sbss, fbhb, pagb
        ! Dependent on imf, sbss, fbhb, delt, dell, pagb, redgb
        imf_type = imf
        pset%imf1 = imf1
        pset%imf2 = imf2
        pset%imf3 = imf3
        pset%vdmc = vdmc
        pset%mdave = mdave
        pset%dell = dell
        pset%delt = delt
        pset%sbss = sbss
        pset%fbhb = fbhb
        pset%pagb = pagb
        
        !set up all SSPs at once (i.e., all metallicities)
        spec_ssp_zz = 0.0
        do zi=1,nz
            ! need to set blue HB and delta AGB before this
            pset%zmet = zi
            call ssp_gen(pset,mass_ssp_zz(zi,:), &
                lbol_ssp_zz(zi,:), spec_ssp_zz(zi,:,:))
        end do
        write (*,*) 'setup_all_ssp() complete'
    end subroutine

    ! TODO directly return outputs from the COMPSPOUT structure
    subroutine comp_sp_file(model_name, dust, zmet, sfh, tau, const, &
            fburst, tburst, dust_tesc, dust1, dust2, dust_clumps, &
            frac_no_dust, dust_index, mwr, wgp1, wgp2, wgp3, &
            duste_gamma, duste_umin, duste_qpah, tage)
        character(len=256) :: output_path
        character(len=256), intent(in) :: model_name
        integer, intent(in) :: dust, zmet, sfh
        real, intent(in) :: tau, const, fburst, tburst, dust_tesc
        real, intent(in) :: dust1, dust2, dust_clumps, frac_no_dust
        real, intent(in) :: dust_index, mwr
        integer, intent(in) :: wgp1, wgp2, wgp3
        real, intent(in) :: duste_gamma, duste_umin, duste_qpah, tage
        dust_type = dust
        pset%zmet = zmet
        pset%sfh = sfh
        pset%tau = tau
        pset%const = const
        pset%tage = tage
        pset%fburst = fburst
        pset%tburst = tburst
        pset%dust_tesc = dust_tesc
        pset%dust1 = dust1
        pset%dust2 = dust2
        pset%dust_clumps = dust_clumps
        pset%frac_nodust = frac_no_dust
        pset%dust_index = dust_index
        pset%mwr = mwr
        pset%wgp1 = wgp1
        pset%wgp2 = wgp2
        pset%wgp3 = wgp3
        pset%duste_gamma = duste_gamma
        pset%duste_umin = duste_umin
        pset%duste_qpah = duste_qpah
        ! For calling COMPSP
        ! depends on zmet, SFH, dust
        ! we want to directly return the results, for now use file output
        output_path = trim(model_name)//'.out'
        call compsp(3, 1, output_path, mass_ssp_zz(zmet,:), &
            lbol_ssp_zz(zmet,:), spec_ssp_zz(zmet,:,:), pset, ocompsp)
        write (*,*) 'comp_sp_file() complete'
    end subroutine

    ! Computes the SP, keeping the results to memory
    ! Call get_sp_mags() or get_sp_spec() to get outputs
    subroutine comp_sp(dust, zmet, sfh, tau, const, &
            fburst, tburst, dust_tesc, dust1, dust2, dust_clumps, &
            frac_no_dust, dust_index, mwr, wgp1, wgp2, wgp3, &
            duste_gamma, duste_umin, duste_qpah, tage)
        character(len=256) :: output_path
        integer, intent(in) :: dust, zmet, sfh
        real, intent(in) :: tau, const, fburst, tburst, dust_tesc
        real, intent(in) :: dust1, dust2, dust_clumps, frac_no_dust
        real, intent(in) :: dust_index, mwr
        integer, intent(in) :: wgp1, wgp2, wgp3
        real, intent(in) :: duste_gamma, duste_umin, duste_qpah, tage
        dust_type = dust
        pset%zmet = zmet
        pset%sfh = sfh
        pset%tau = tau
        pset%const = const
        pset%tage = tage
        pset%fburst = fburst
        pset%tburst = tburst
        pset%dust_tesc = dust_tesc
        pset%dust1 = dust1
        pset%dust2 = dust2
        pset%dust_clumps = dust_clumps
        pset%frac_nodust = frac_no_dust
        pset%dust_index = dust_index
        pset%mwr = mwr
        pset%wgp1 = wgp1
        pset%wgp2 = wgp2
        pset%wgp3 = wgp3
        pset%duste_gamma = duste_gamma
        pset%duste_umin = duste_umin
        pset%duste_qpah = duste_qpah
        ! For calling COMPSP
        ! depends on zmet, SFH, dust
        output_path = "foo"
        call compsp(0, 1, output_path, mass_ssp_zz(zmet,:), &
            lbol_ssp_zz(zmet,:), spec_ssp_zz(zmet,:,:), pset, ocompsp)
        write (*,*) 'comp_sp() complete'
    end subroutine

    ! Returns number of filters that mags are computed for
    subroutine get_n_bands(n_bands)
        integer, intent(out) :: n_bands
        n_bands = nbands
    end subroutine

    ! Returns number of ages that are computed in a SFH, if tage is set to 0.
    subroutine get_n_ages(n_ages)
        integer, intent(out) :: n_ages
        n_ages = ntfull
    end subroutine

    ! Returns number of masses included in the isochrones
    subroutine get_n_masses(n_masses)
        integer, intent(out) :: n_masses
        n_masses = nm
    end subroutine

    ! Get mags for a single age given by index iage
    ! todo figure out how this interacts with setting tage; does ocompsp
    ! simply place all data at the first index? Or at the proper index?
    ! Or is tage only related to writing output files?
    subroutine get_mags_at_age(iage, n_bands, mag_array)
        integer, intent(in) :: iage, n_bands
        real, dimension(n_bands), intent(out) :: mag_array
        write (*,*) 'in get_mags_at_age'
        do zi=1,n_bands
            write (*,*) ocompsp(iage)%mags(zi)
        end do
        mag_array = ocompsp(iage)%mags
        write (*,*) 'copied mag_array'
    end subroutine

    ! Get mags for all ages
    subroutine get_mags(n_bands, n_ages, mag_array)
        integer, intent(in) :: n_bands, n_ages
        real, dimension(n_bands,n_ages), intent(out) :: mag_array
        do zi=1,n_ages
            mag_array(:,zi) = ocompsp(zi)%mags
        end do
    end subroutine

end module
