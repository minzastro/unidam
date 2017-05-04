module model_fitter
! Fortran module of the SAGE_GAP distance estimation tool
implicit none

real, allocatable :: models(:, :)
logical, allocatable :: mask_models(:)
real, allocatable :: model_params(:, :)
real, allocatable :: param(:), param_err(:)
real, allocatable :: mag(:), mag_err(:), Ck(:)
integer, allocatable :: abs_mag(:), model_columns(:), fitted_columns(:)
integer, save :: model_column_count
real, save :: matrix0(2, 2), matrix_det
real, save :: max_param_err = 4.
logical, save :: use_model_weight = .true.
logical, save :: use_magnitude_probability = .true.
logical, save :: debug = .false.
logical, save :: allow_negative_extinction = .false.

! Flag indicating if the distance is known
logical, save :: distance_known = .false.
logical, save :: parallax_known = .false.
real, save :: distance_modulus, distance_modulus_err
real, save :: parallax, parallax_error, extinction, extinction_error

! Distance prior:
! 0 = none
! 1 = d^2
! 2 = RAVE Galaxy model (unimplemented)
! 3 = RAVE Galaxy model + d^2 (unimplemented)
integer, save :: distance_prior = 1

contains

subroutine alloc_settings(na, xabs_mag, nmod, xmodel_columns, nf, xfitted_columns)
  integer, intent(in) :: na, nmod, nf
  integer, intent(in) :: xabs_mag(na), xmodel_columns(nmod), xfitted_columns(nf)
    if (allocated(abs_mag)) then
        deallocate(abs_mag)
    endif
    allocate(abs_mag(na))
    abs_mag = xabs_mag + 1
    if (allocated(model_columns)) then
        deallocate(model_columns)
    endif
    allocate(model_columns(nmod))
    model_columns = xmodel_columns + 1
    if (allocated(fitted_columns)) then
        deallocate(fitted_columns)
    endif
    allocate(fitted_columns(nf))
    fitted_columns = xfitted_columns + 1
end subroutine alloc_settings

subroutine alloc_models(nn, mm, modarray) ! Load models
  integer, intent(in) :: nn, mm
  real, intent(in) :: modarray(nn, mm)
    if (allocated(models)) then
        deallocate(models)
        deallocate(mask_models)
    endif
    allocate(models(nn, mm))
    allocate(mask_models(nn))
    mask_models = .true.
    models = modarray
    model_column_count = mm
end subroutine alloc_models

subroutine alloc_param(n, xparam, xparam_err) ! Load data for model params
  integer, intent(in) :: n
  real, intent(in) :: xparam(n), xparam_err(n)
    if (allocated(param)) then
        deallocate(param)
    endif
    allocate(param(n))
    param = xparam
    if (allocated(param_err)) then
        deallocate(param_err)
    endif
    allocate(param_err(n))
    param_err = xparam_err
end subroutine alloc_param

subroutine alloc_mag(n, xmag, xmag_err, xCk) ! Load data for magnitudes
  integer, intent(in) :: n
  real, intent(in) :: xmag(n), xmag_err(n), xCk(n)
    if (allocated(mag)) then
        deallocate(mag)
    endif
    allocate(mag(n))
    mag = xmag
    if (allocated(mag_err)) then
        deallocate(mag_err)
    endif
    allocate(mag_err(n))
    mag_err = xmag_err
    if (allocated(Ck)) then
        deallocate(Ck)
    endif
    allocate(Ck(n))
    Ck = xCk
end subroutine alloc_mag

subroutine solve_for_distance(vector, solution)
  real, intent(in) :: vector(2)
  real, intent(out) :: solution(2)
    solution(1) = (vector(1)*matrix0(2,2) - vector(2)*matrix0(1,2))*matrix_det
    solution(2) = -(vector(1)*matrix0(1,2) - vector(2)*matrix0(1,1))*matrix_det
end subroutine solve_for_distance

real function mu_d_derivative(ext, mu, vector)
  ! Derivative of L_sed with mu_d, with volume correction AND known parallax
  real, intent(in) :: ext, mu
  real, intent(in) :: vector(2)
  real pi
    pi = 10**(-0.2 * mu - 1.)
    mu_d_derivative = -0.2 * log(10.) * (2. + pi * (pi - parallax) / parallax_error**2) - vector(1) + sum((ext * Ck + mu) * mag_err)
end function mu_d_derivative

real function extinction_derivative(ext, mu, vector)
  real, intent(in) :: ext, mu
  real, intent(in) :: vector(2)
    extinction_derivative = (ext - extinction) / extinction_error**2 - &
      vector(2) + sum((ext * Ck + mu) * Ck * mag_err)
end function extinction_derivative

real function mu_d_function(mud, vector)
  real, intent(in) :: mud
  real, intent(in) :: vector(2)
  real Ak_local, pi
      pi = 10**(-0.2 * mud - 1.)
      Ak_local = ((extinction / extinction_error**2) + vector(2) - &
                   sum(Ck * mud * mag_err)) / (sum(Ck * Ck * mag_err) + (1/extinction_error**2))
      mu_d_function = 0.2 * log(10.) * (2. + pi * (pi - parallax)/ parallax_error**2) + &
        vector(1) - sum((Ak_local * Ck + mud) * mag_err)
      !write(97, *) mud, pi, Ak_local, extinction
end function mu_d_function

subroutine solve_for_distance_with_parallax(vector, solution)
  real, intent(in) :: vector(2)
  real, intent(inout) :: solution(2)
  real Ak_old, mu_old
  real Ak_new, mu
  real mu_1, mu_2, fun_1, fun_2
  integer iterations
    !write(99, *) solution, parallax, parallax_error, extinction, extinction_error
    !write(98, *) '---------------'
    mu_1 = -5 * (1. + log10(parallax + 10.*parallax_error))
    if (parallax .gt. 10.*parallax_error) then
        mu_2 = -5 * (1. + log10(parallax - 10.*parallax_error))
    else
        mu_2 = -5 * (1. + log10(parallax)) - 5.
    endif
    iterations = 0
    !write(98, *) mu_1, mu_2
    do while ((iterations .le. 100) .and. (abs(mu_2 - mu_1) .gt. 1e-6))
      fun_1 = mu_d_function(mu_1, vector)
      fun_2 = mu_d_function(mu_2, vector)
      if (fun_1 * fun_2 .le. 0) then
        if (abs(fun_1) .gt. abs(fun_2)) then
          mu_1 = mu_1 + 0.5 * (mu_2 - mu_1)
        else
          mu_2 = mu_2 - 0.5 * (mu_2 - mu_1)
        endif
      else if (abs(fun_1) .gt. abs(fun_2)) then
        mu_1 = mu_2 + 2.5*(mu_2 - mu_1)
      else
        mu_2 = mu_1 + 2.5*(mu_1 - mu_2)
      endif
      !write(98, *) mu_1, mu_2, fun_1, fun_2
      iterations = iterations + 1
    enddo
    solution(1) = mu_1
    solution(2) = ((extinction / extinction_error**2) + vector(2) - &
                  sum(Ck * mu_1 * mag_err)) / (sum(Ck * Ck * mag_err) + (1/extinction_error**2))
end subroutine solve_for_distance_with_parallax


subroutine find_best(m_count)
  ! Finding stellar parameters from observed + models
  integer, parameter :: WSIZE = 30 ! Total number of parameters for output
  integer, intent(out) :: m_count
  integer i
  integer off, prob
  real p
  real L_model, L_sed, L_sednoext, bic2, bic1
  real vector(2)
  real mu_d(2), mu_d_noext ! (mu_d, Av)
    if (allocated(model_params)) then
      deallocate(model_params)
    endif
    allocate(model_params(size(mask_models), WSIZE))
    ! First we filter out models by *max_param_err* times sigma
    do i = 1, size(param)
      mask_models = mask_models .and. (abs(models(:, model_columns(i)) - param(i)).le.(max_param_err*param_err(i)))
    enddo
    off = size(fitted_columns)
    prob = off + 5 ! Here probablities start
    m_count = 1
    if (distance_known) then
      matrix0(1, 1) = matrix0(1, 1) + 1./distance_modulus_err**2
      matrix0(2, 2) = matrix0(2, 2) + 1./(25.*extinction**2)
      matrix_det = 1./(matrix0(1, 1) * matrix0(2, 2) - matrix0(1, 2)*matrix0(2, 1))
    endif
    do i = 1, size(mask_models)
      if (mask_models(i)) then ! Take only filtered models
        ! Calculate chi^2 value for model parameters:
        L_model = 0.5*sum(((models(i, model_columns) - param) / param_err)**2)
        if (L_model.ge.0.5*max_param_err**2) then
          ! Further filtering - replace box clipping by a circle
          mask_models(i) = .false.
          cycle
        endif
        mu_d(:) = -1.
        if (distance_known) then
          ! Distance is known, solve for extinction only
          vector(1) = sum((mag - models(i, abs_mag))*mag_err) + distance_modulus / distance_modulus_err**2
          vector(2) = sum((mag - models(i, abs_mag))*mag_err*Ck) + 1./(25.*extinction)
          mu_d_noext = vector(1) / matrix0(1, 1)
          L_sednoext = 0.5*(sum((mag - mu_d_noext - models(i, abs_mag))**2 * mag_err) + &
                               ((mu_d_noext - distance_modulus)/distance_modulus_err)**2 + &
                               (1./25.)) ! this is for extinction, sigma_A = 5 A_0
        else if (parallax_known) then
          vector(1) = sum((mag - models(i, abs_mag))*mag_err)
          vector(2) = sum((mag - models(i, abs_mag))*mag_err*Ck)
          mu_d_noext = vector(1) / matrix0(1, 1)
          L_sednoext = 0.5*sum((mag - mu_d_noext - models(i, abs_mag))**2 * mag_err)
        else ! No distance or parallax known
          vector(1) = sum((mag - models(i, abs_mag))*mag_err)
          if (distance_prior.eq.1) then
            vector(1) = vector(1) + log(10.)*0.4
          endif
          vector(2) = sum((mag - models(i, abs_mag))*mag_err*Ck)
          mu_d_noext = vector(1) / matrix0(1, 1)
          L_sednoext = 0.5*sum((mag - mu_d_noext - models(i, abs_mag))**2 * mag_err)
        endif
        if ((size(mag_err).ge.2).or.(parallax_known)) then
          if (parallax_known) then
              bic1 = 1e10
              mu_d(1) = -5. * (log10(parallax) + 1.)
              mu_d(2) = extinction
              call solve_for_distance_with_parallax(vector, mu_d)
              L_sed = 0.5*sum((mag - mu_d(1) - models(i, abs_mag) - Ck * mu_d(2))**2 * mag_err)
          else
              bic1 = 2.*L_sednoext + log(float(size(mag_err)))
              call solve_for_distance(vector, mu_d)
              if (distance_known) then
                L_sed = 0.5*sum((mag - mu_d(1) - models(i, abs_mag) - Ck * mu_d(2))**2 * mag_err) + &
                        0.5 * (mu_d(2) - extinction)**2 / (25.*extinction**2) + &
                        0.5 * (mu_d(1) - distance_modulus)**2 / distance_modulus_err**2
              else
                L_sed = 0.5*sum((mag - mu_d(1) - models(i, abs_mag) - Ck * mu_d(2))**2 * mag_err)
              endif
          endif
          bic2 = 2.*L_sed + 2.*log(float(size(mag_err)))
          if (((mu_d(2).lt.0.0).and.(.not.allow_negative_extinction)) .or. &
              (bic1.le.bic2)) then
            ! Negative extinction. This is not physical - replace it by 0 and recalculate dm
            ! OR we get a better fit with no assumption on extinction at all.
            L_sed = L_sednoext
            mu_d(1) = mu_d_noext
            mu_d(2) = 0d0
          endif
        else
          L_sed = L_sednoext
          mu_d(1) = mu_d_noext
          mu_d(2) = 0d0
        endif
        model_params(m_count, :off) = models(i, fitted_columns)
        ! Distance modulus + Extinction
        model_params(m_count, off+1:off+2) = mu_d
        ! Distance
        model_params(m_count, off+3) = 10**(mu_d(1)*0.2 + 1)
        ! Parallax
        model_params(m_count, off+4) = 1./model_params(m_count, off+3)
        ! Unweighted isochrone likelihood
        if (parallax_known) then
            L_sed = L_sed +  & !0.5 * (mu_d(2) - extinction)**2 / (extinction_error**2) + &
                            0.5 * (model_params(m_count, off+4) - parallax)**2 / (parallax_error**2)
        endif
        model_params(m_count, prob) = L_model
        ! SED likelihood
        model_params(m_count, prob+1) = L_sed
        if (use_magnitude_probability) then
           ! Multiply chi2 by residuals of SED fit
           p = exp(-L_model - L_sed)
        else
           p = exp(-L_model)
        endif
        if (isnan(L_sed).or.(p .le. epsilon(1.))) then
		  if (debug) then
		    write(68, *) models(i, model_columns), models(i, abs_mag), &
			             'Mag:', mag - models(i, abs_mag), &
			             'Fit:', model_params(m_count, 1:off+4), L_model, L_sed, p
		  endif
          mask_models(i) = .false.
          cycle
        endif
        if (use_model_weight) then
           ! Multiply by model weight (combined of age and mass weighting)
           p = p * models(i, model_column_count)
        endif
        if ((distance_prior.eq.1).or.(parallax_known)) then
           ! Multiply by d^2 - volume factor correction
           p = p * model_params(m_count, off+3) * model_params(m_count, off+3)
        endif
        model_params(m_count, prob+2) = p
        if (debug) then
          write(66, *) models(i, model_column_count), models(i, abs_mag), &
                       models(i, model_columns), models(i, model_column_count-1), -99, &
                       model_params(m_count, 1:prob+2), mag - models(i, abs_mag), L_model, L_sed, p
        endif
        m_count = m_count + 1
      endif
    enddo
    m_count = m_count - 1
    !model_params(m_count+1:, prob+2) = -1
end subroutine find_best

end module model_fitter