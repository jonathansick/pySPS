! fsps.f90 -- a module intended to be wrapped by f2py as a python module
! for making stellar pops

module driver
    use sps_vars; use nrtype; use sps_utils
    implicit none

    type(params) :: pset
    !f2py intent(hide) pset
    type(COMPSPOUT), dimension(ntfull) :: ocompsp
    !f2py intent(hide) ocompsp
    integer :: zi

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

    subroutine setup_one_ssp(zz, imf, imf1, imf2, imf3, vdmc, &
            mdave, dell, delt, sbss, fbhb, pagb)
        integer, intent(in) :: zz, imf
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
        
        ! Initialize just one metallicity, index zi
        spec_ssp_zz = 0.0
        ! need to set blue HB and delta AGB before this
        pset%zmet = zz
        call ssp_gen(pset,mass_ssp_zz(zz,:), &
            lbol_ssp_zz(zz,:), spec_ssp_zz(zz,:,:))
        write (*,*) 'setup_one_ssp() complete'
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

    ! Returns number of wavelength bins in spectra
    subroutine get_n_lambda(n_lambda)
        integer, intent(out) :: n_lambda
        n_lambda = nspec
    end subroutine

    ! Returns number of ages that are computed in a SFH, if tage is set to 0.
    ! Use this with comp sp
    subroutine get_n_ages(n_ages)
        integer, intent(out) :: n_ages
        n_ages = ntfull
    end subroutine

    ! Returns number of ages in the basic isochrone set
    ! use this with isochrones
    subroutine get_n_ages_isochrone(n_ages)
        integer, intent(out) :: n_ages
        n_ages = nt
    end subroutine

    ! Returns max number of masses included in the isochrones
    subroutine get_n_masses(n_masses)
        integer, intent(out) :: n_masses
        n_masses = nm
    end subroutine

    ! Returns number of masses included in a specific isochrone
    ! Use this with the get_isochrone() function
    subroutine get_n_masses_isochrone(zz, tt, nmass)
        integer, intent(in) :: zz, tt
        integer, intent(out) :: nmass
        nmass = nmass_isoc(zz,tt)
    end subroutine

    ! Returns array with spectral grid
    subroutine get_lambda_grid(n_lambda, lambda_array)
        integer, intent(in) :: n_lambda
        real, dimension(n_lambda), intent(out) :: lambda_array
        lambda_array = spec_lambda
    end subroutine

    ! Get mags for a single age given by index iage
    ! todo figure out how this interacts with setting tage; does ocompsp
    ! simply place all data at the first index? Or at the proper index?
    ! Or is tage only related to writing output files?
    subroutine get_csp_mags_at_age(iage, n_bands, mag_array)
        integer, intent(in) :: iage, n_bands
        real, dimension(n_bands), intent(out) :: mag_array
        write (*,*) 'in get_mags_at_age'
        do zi=1,n_bands
            write (*,*) ocompsp(iage)%mags(zi)
        end do
        mag_array = ocompsp(iage)%mags
        write (*,*) 'copied mag_array'
    end subroutine

    ! Get the age, mass, bolometric luminosity, SFR and dust mass at
    ! a single age
    subroutine get_csp_stats_at_age(iage, age, mass, lbol, sfr, dust_mass)
        integer, intent(in) :: iage
        real, intent(out) :: age, mass, lbol, sfr, dust_mass
        age = ocompsp(iage)%age
        mass = ocompsp(iage)%mass_csp
        lbol = ocompsp(iage)%lbol_csp
        sfr = ocompsp(iage)%sfr
        dust_mass = ocompsp(iage)%mdust
    end subroutine

    ! Get mags for all ages
    subroutine get_csp_mags(n_bands, n_ages, mag_array)
        integer, intent(in) :: n_bands, n_ages
        real, dimension(n_ages,n_bands), intent(out) :: mag_array
        do zi=1,n_ages
            mag_array(zi,:) = ocompsp(zi)%mags
        end do
    end subroutine

    ! Get spectra for all ages
    subroutine get_csp_specs(n_lambda, n_ages, spec_array)
        integer, intent(in) :: n_lambda, n_ages
        real, dimension(n_ages,n_lambda), intent(out) :: spec_array
        do zi=1,n_ages
            spec_array(zi,:) = ocompsp(zi)%spec
        end do
    end subroutine

    ! Get spectra for a single indexed age
    subroutine get_csp_specs_at_age(iage, n_lambda, spec_array)
        integer, intent(in) :: iage, n_lambda
        real, dimension(n_lambda), intent(out) :: spec_array
        spec_array = ocompsp(iage)%spec
    end subroutine

    ! Get stellar pop statistics for all ages
    subroutine get_csp_stats(n_ages, age, mass, lbol, sfr, dust_mass)
        integer, intent(in) :: n_ages
        real, dimension(n_ages), intent(out) :: age, mass, lbol, sfr, dust_mass
        do zi=1,n_ages
            age(zi) = ocompsp(zi)%age
            mass(zi) = ocompsp(zi)%mass_csp
            lbol(zi) = ocompsp(zi)%lbol_csp
            sfr(zi) = ocompsp(zi)%sfr
            dust_mass(zi) = ocompsp(zi)%mdust
        end do
    end subroutine

    ! Returns the isochrone for a given metallicity and age index
    subroutine get_isochrone(zz, tt, n_mass, n_mags, time_out, z_out, &
            mass_init_out, logl_out, logt_out, logg_out, ffco_out, &
            phase_out, wght_out, mags_out)
        integer, intent(in) :: zz, tt, n_mass, n_mags
        real, intent(out) :: time_out, z_out
        real, dimension(n_mass), intent(out) :: mass_init_out
        real, dimension(n_mass), intent(out) :: logl_out
        real, dimension(n_mass), intent(out) :: logt_out
        real, dimension(n_mass), intent(out) :: logg_out
        real, dimension(n_mass), intent(out) :: ffco_out ! C or M-type AGB
        real, dimension(n_mass), intent(out) :: phase_out ! TODO how to decode
        real, dimension(n_mass), intent(out) :: wght_out
        real, dimension(n_mass, n_mags), intent(out) :: mags_out
        integer :: i
        real, dimension(nm) :: wght ! 1500; max n of masses for any isoc
        !f2py intent(hide) wght
        real, dimension(nspec)  :: spec
        !f2py intent(hide) spec
        real, dimension(nbands) :: mags ! vector of mags for FSPS
        !f2py intent(hide) mags
        
        ! check that n_mass == nmass_isoc(zz,tt)
        !write (*,*) 'In get_isochrone()'
        call IMF_WEIGHT(mini_isoc(zz,tt,:), wght, nmass_isoc(zz,tt))
        !write (*,*) 'Finished IMF_WEIGHT'
        !write (*,*) nmass_isoc(zz,tt)
        do i = 1, nmass_isoc(zz,tt)
            ! Compute mags on isochrone at this mass
            call GETSPEC(zz, mini_isoc(zz,tt,i), mact_isoc(zz,tt,i), &
                    logt_isoc(zz,tt,i), 10**logl_isoc(zz,tt,i), &
                    phase_isoc(zz,tt,i), ffco_isoc(zz,tt,i), spec)
            call GETMAGS(0.0, spec, mags)
            ! Fill in outputs for this mass
            !write (*,*) i
            !write (*,*) logl_isoc(zz,tt,i)
            !write (*,*) ffco_isoc(zz,tt,i)
            !write (*,*) phase_isoc(zz,tt,i)
            !write (*,*) wght(i)
            mass_init_out(i) = mini_isoc(zz,tt,i)
            logl_out(i) = logl_isoc(zz,tt,i)
            logt_out(i) = logt_isoc(zz,tt,i)
            logg_out(i) = logg_isoc(zz,tt,i)
            ffco_out(i) = ffco_isoc(zz,tt,i)
            phase_out(i) = phase_isoc(zz,tt,i)
            wght_out(i) = wght(i)
            mags_out(i,:) = mags(:)
            !write (*,*) mags_out(i,:)
            !write (*,*) '--'
        end do

        ! Fill in time and metallicity of this isochrone
        time_out = timestep_isoc(zz, tt)
        z_out = LOG10(zlegend(zz) / 0.0190) ! log(Z/Zsolar)
    end subroutine


end module
