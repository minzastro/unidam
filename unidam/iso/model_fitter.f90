module model_fitter
!! Fortran module of the UniDAM.
!! Equations referred to are from Mints and Hekker (2017).
use Solve_NonLin
implicit none

!> All models
real, allocatable :: models(:, :)

!> This array contains a mask for models within 4 sigmas
logical, allocatable :: mask_models(:)

!> Special array of flags.
!> Indicates if the following columns are needed:
!> Distance modulus, extinction, distance, parallax
logical, save :: special_columns(4)

!> Parameters of the selected models
!> First *size(fitted_columns)* columns are filled with
!> model observables (Teff, logg, feh in the 'classical' case)
!> Next 0 to 4 columns with
!> distance modulus, extinction, distance, parallax
!> are added depending on *special_columns* flag array values
!> Next 4 columns are L_model, L_sed, p and model index
real, allocatable :: model_params(:, :)
!> Observed parameters for a current star (T, logg, feh)
real, allocatable :: param(:)
!> Uncertainties of parameters for a current star (T, logg, feh)
real, allocatable :: param_err(:)

!> Wether to add L_sed to L_model
logical, save :: use_photometry = .true.
!> Visible magnitudes for a current star
real, allocatable :: mag(:)
!> Uncertainties of visible magnitudes for a current star
real, allocatable :: mag_err(:)
!> Extinction coefficiens (see eq 8 in Paper 1)
real, allocatable :: Ck(:)
!> Indices of absolute magnitude columns in models array 
integer, allocatable :: abs_mag(:)
!> Indices of "observed" columns in models array (default: T, logg, feh) 
integer, allocatable :: model_columns(:)
!> Indices of columns for derived values in models array 
integer, allocatable :: fitted_columns(:)
!> Number of columns in the models array
integer, save :: model_column_count
!> Matrix for eq 15 and inverse of its determinant
real, save :: matrix0(2, 2)
!> Determinant of matrix0
real, save :: matrix_det
!> Maximum deviation from param(:) for models
real, save :: max_param_err = 4.
!> Use mass, age and metallicity weighting for models
logical, save :: use_model_weight = .true.
!> Use SED fit to constrain models
logical, save :: use_magnitude_probability = .true.
!> Output debugging information
logical, save :: debug = .false.
!> Allow for negative extinctions when solving for distances
logical, save :: allow_negative_extinction = .false.

!> Flag indicating if the parallax is known
logical, save :: parallax_known = .false.
real, save :: parallax, parallax_error, extinction, extinction_error
real, save :: parallax_L_correction
!> Distance prior:
!> 0 = none
!> 1 = d^2
!> 2 = RAVE Galaxy model (unimplemented)
!> 3 = RAVE Galaxy model + d^2 (unimplemented)
integer, save :: distance_prior = 1
real, save :: prior_parameter = 2e3

contains

! ALLOCATION ROUTINES
!> Allocate settings from config file
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

subroutine alloc_models(nn, mm, modarray)
!! Load models
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

subroutine alloc_param(n, xparam, xparam_err)
!! Load data for model params
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

subroutine alloc_mag(n, xmag, xmag_err, xCk)
!! Load data for magnitudes
  !> Number of magnitudes
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


!! Other routines
subroutine solve_for_distance(vector, solution)
  real, intent(in) :: vector(2)
  real, intent(out) :: solution(2)
    solution(1) = (vector(1)*matrix0(2,2) - vector(2)*matrix0(1,2))*matrix_det
    solution(2) = -(vector(1)*matrix0(1,2) - vector(2)*matrix0(1,1))*matrix_det
end subroutine solve_for_distance


subroutine solve_for_distance_with_parallax(vector, solution)
  !! Solving the system of equations for distance modulus
  !! and extinction for the case with parallax and extinction priors.
  !! This is probably highly inefficient, an improvement
  !! is needed here.
  real, intent(in) :: vector(2)
  real, intent(inout) :: solution(2)
  integer :: info
  real :: tol=1e-8
  real, dimension(2) :: fvec, diag
    call hbrd(fcnx, 2, solution, fvec, epsilon(tol), tol, info, diag)
    return
    contains
    SUBROUTINE FCNX(N, X, FVEC, IFLAG)
        IMPLICIT NONE
        INTEGER, INTENT(IN)      :: n
        REAL, INTENT(IN)    :: x(n)
        REAL, INTENT(OUT)   :: fvec(n)
        INTEGER, INTENT(IN OUT)  :: iflag
        real extra, pi
          pi = 1./mu_d_to_distance(x(1))
          if (X(2) .ge. extinction) then
            extra = (X(2) - extinction) / extinction_error**2
          else
            extra = 0
          endif
          FVEC(1) = -vector(1) + sum(Ck * X(2) * mag_err) + &
            sum(X(1) * mag_err) - &
            0.2 * log(10.) * pi * (pi - parallax)/ parallax_error**2
          if (distance_prior.eq.1) then
            FVEC(1) = FVEC(1) + 0.4 * log(10.)
          else if (distance_prior.eq.2) then
            FVEC(1) = FVEC(1) + 0.2 * log(10.) * (2. - 1./(pi * prior_parameter)) 
          endif
          FVEC(2) = -vector(2) + sum(Ck * Ck * X(2) * mag_err) + &
            sum(Ck * X(1) * mag_err) + extra
    END SUBROUTINE FCNX
end subroutine solve_for_distance_with_parallax


subroutine get_vector(model_mags, vector, L_sednoext, mu_d_noext)
!! Get vector for mu_d/extinction system of eq.
!! and L_sednoext = L_sed with zero extinction.
  !integer, intent(in) ::
  real, intent(in) :: model_mags(:)
  real, intent(out) :: L_sednoext 
  real, intent(out) :: vector(2)
  real, intent(out) :: mu_d_noext
    if (parallax_known) then
      vector(1) = sum((mag - model_mags)*mag_err)
      vector(2) = sum((mag - model_mags)*mag_err*Ck)
      mu_d_noext = vector(1) / matrix0(1, 1)
      L_sednoext = 0.5 * sum((mag - mu_d_noext - model_mags)**2 * mag_err) + &
                   0.5 * (1. / mu_d_to_distance(mu_d_noext) - parallax)**2 / (parallax_error**2) + &
                   parallax_L_correction
    else ! No distance or parallax known
      vector(1) = sum((mag - model_mags)*mag_err)
      if (distance_prior.eq.1) then
        vector(1) = vector(1) + log(10.)*0.4
      else if (distance_prior.eq.2) then
        vector(1) = vector(1) + 0.2 * log(10.) * (2. - mu_d_to_distance(mu_d_noext)/prior_parameter) 
      endif
      vector(2) = sum((mag - model_mags)*mag_err*Ck)
      mu_d_noext = vector(1) / matrix0(1, 1)
      L_sednoext = 0.5 * sum((mag - mu_d_noext - model_mags)**2 * mag_err)
    endif
end subroutine get_vector

real function mu_d_to_distance(mu)
  !! Convert distance modulus to distance
  real, intent(in) :: mu
    mu_d_to_distance = 10**(mu*0.2 + 1)
end function mu_d_to_distance

subroutine process_model(model, out_size, success, out_model)
  real, intent(in) :: model(:)
  integer, intent(in) :: out_size
  logical, intent(out) :: success
  real, intent(out) :: out_model(out_size)
  integer i
  integer off, prob
  real p, distance
  real L_model, L_sed, bic2, bic1
  real L_sednoext, mu_d_noext ! L_sed and mu_d if extinction == zero
  real vector(2)
  real mu_d(2) ! (mu_d, Av)
    success = .true.
    prob = size(fitted_columns) + count(special_columns) + 1 ! Here probablities start
    ! Calculate chi^2 value for model parameters:
    L_model = 0.5*sum(((model(model_columns) - param) / param_err)**2)
    if (L_model.ge.0.5*max_param_err**2) then
      ! Further filtering - replace box clipping by a circle
      success = .false.
      return
    endif
    mu_d(:) = -1.
    off = size(fitted_columns)
    out_model(:off) = model(fitted_columns)
    if (use_photometry) then
        call get_vector(model(abs_mag), vector, L_sednoext, mu_d_noext)
        if ((size(mag_err).ge.2).or.(parallax_known)) then
          if (parallax_known) then
              bic1 = 1e10
              ! This is just first guess, so abs(parallax) is Ok.
              mu_d(1) = -5. * (log10(abs(parallax)) + 1.)
              mu_d(2) = extinction
              call solve_for_distance_with_parallax(vector, mu_d)
              L_sed = 0.5*sum((mag - mu_d(1) - model(abs_mag) - Ck * mu_d(2))**2 * mag_err)
              L_sed = L_sed + 0.5 * (1. / mu_d_to_distance(mu_d(1)) - parallax)**2 / (parallax_error**2)
              if (mu_d(2) .ge. extinction) then
                  L_sed = L_sed + 0.5 * (mu_d(2) - extinction)**2 / (extinction_error**2)
              endif
              L_sed = L_sed + parallax_L_correction
          else
              ! If there are 2 or more bands observed
              ! then we can solve eq. 15
              bic1 = 2.*L_sednoext + log(float(size(mag_err)))
              call solve_for_distance(vector, mu_d)
              L_sed = 0.5*sum((mag - mu_d(1) - model(abs_mag) - Ck * mu_d(2))**2 * mag_err)
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
        ! Distance modulus
        if (special_columns(1)) then
            off = off + 1
            out_model(off) = mu_d(1)
        endif
        ! Extinction
        if (special_columns(2)) then
            off = off + 1
            out_model(off) = mu_d(2)
        endif
        ! Distance
        distance = mu_d_to_distance(mu_d(1))
        if (special_columns(3)) then
            off = off + 1
            out_model(off) = distance
        endif
        ! Parallax
        if (special_columns(4)) then
            off = off + 1
            out_model(off) = 1. / distance
        endif
    else
        L_sed = 0
    endif
    out_model(prob) = L_model
    ! SED likelihood
    out_model(prob+1) = L_sed
    if (use_magnitude_probability) then
       ! Multiply chi2 by residuals of SED fit
       p = exp(-L_model - L_sed)
    else
       p = exp(-L_model)
    endif
    if (isnan(L_sed).or.(p .le. tiny(1.))) then
      if (debug) then
        write(68, *) model(model_columns), model(abs_mag), &
                     'Mag:', mag - model(abs_mag), &
                     'Fit:', out_model(1:off), L_model, L_sed, p, i
      endif
      success = .false.
      return
    endif
    if (use_model_weight) then
      ! Multiply by model weight (combined of age and mass weighting)
      p = p * model(model_column_count)
    endif
    if (use_photometry) then
      ! Prevent the use of distance prior when no photometry
      ! is used and therefore no distance is estimated.
      if (distance_prior.eq.1) then
        ! Multiply by d^2 - volume factor correction
        p = p * distance * distance
      else if (distance_prior.eq.2) then
        p = p * (distance * distance * exp(-distance / prior_parameter))
      endif
    endif
    out_model(prob+2) = p
    out_model(prob+3) = i
    if (debug) then
      write(66, *) model(model_column_count), model(abs_mag), &
                   model(model_columns), model(model_column_count-1), -99, &
                   out_model(1:prob+2), mag - model(abs_mag), L_model, L_sed, p, i, distance_prior
    endif
end subroutine process_model

subroutine find_best(m_count)
  !! Finding stellar parameters from observed + models
  !> Maximum number of parameters for output
  integer, parameter :: WSIZE = 30
  !> Number of consistent models found.
  integer, intent(out) :: m_count
  integer i
  integer off, prob
  real p, distance
  real L_model, L_sed, bic2, bic1
  real L_sednoext, mu_d_noext ! L_sed and mu_d if extinction == zero
  real vector(2)
  real mu_d(2) ! (mu_d, Av)
    if (allocated(model_params)) then
      deallocate(model_params)
    endif
    ! First we filter out models by *max_param_err* times sigma
    do i = 1, size(param)
      mask_models = mask_models .and. (abs(models(:, model_columns(i)) - param(i)).le.(max_param_err*param_err(i)))
    enddo
    prob = size(fitted_columns) + count(special_columns) + 1 ! Here probablities start
    allocate(model_params(size(mask_models), prob + 3))
    m_count = 1
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
        off = size(fitted_columns)
        model_params(m_count, :off) = models(i, fitted_columns)
        if (use_photometry) then
            call get_vector(models(i, abs_mag), vector, L_sednoext, mu_d_noext)
            if ((size(mag_err).ge.2).or.(parallax_known)) then
              if (parallax_known) then
                  bic1 = 1e10
                  ! This is just first guess, so abs(parallax) is Ok.
                  mu_d(1) = -5. * (log10(abs(parallax)) + 1.)
                  mu_d(2) = extinction
                  call solve_for_distance_with_parallax(vector, mu_d)
                  L_sed = 0.5*sum((mag - mu_d(1) - models(i, abs_mag) - Ck * mu_d(2))**2 * mag_err)
                  L_sed = L_sed + 0.5 * (1. / mu_d_to_distance(mu_d(1)) - parallax)**2 / (parallax_error**2)
                  if (mu_d(2) .ge. extinction) then
                      L_sed = L_sed + 0.5 * (mu_d(2) - extinction)**2 / (extinction_error**2)
                  endif
                  L_sed = L_sed + parallax_L_correction
              else
                  ! If there are 2 or more bands observed
                  ! then we can solve eq. 15
                  bic1 = 2.*L_sednoext + log(float(size(mag_err)))
                  call solve_for_distance(vector, mu_d)
                  L_sed = 0.5*sum((mag - mu_d(1) - models(i, abs_mag) - Ck * mu_d(2))**2 * mag_err)
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
            ! Distance modulus
            if (special_columns(1)) then
                off = off + 1
                model_params(m_count, off) = mu_d(1)
            endif
            ! Extinction
            if (special_columns(2)) then
                off = off + 1
                model_params(m_count, off) = mu_d(2)
            endif
            ! Distance
            distance = mu_d_to_distance(mu_d(1))
            if (special_columns(3)) then
                off = off + 1
                model_params(m_count, off) = distance
            endif
            ! Parallax
            if (special_columns(4)) then
                off = off + 1
                model_params(m_count, off) = 1. / distance
            endif
        else
            L_sed = 0
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
        if (isnan(L_sed).or.(p .le. tiny(1.))) then
          if (debug) then
            write(68, *) models(i, model_columns), models(i, abs_mag), &
                         'Mag:', mag - models(i, abs_mag), &
                         'Fit:', model_params(m_count, 1:off), L_model, L_sed, p, i
          endif
          mask_models(i) = .false.
          cycle
        endif
        if (use_model_weight) then
          ! Multiply by model weight (combined of age and mass weighting)
          p = p * models(i, model_column_count)
        endif
        if (use_photometry) then
          ! Prevent the use of distance prior when no photometry
          ! is used and therefore no distance is estimated.
          if (distance_prior.eq.1) then
            ! Multiply by d^2 - volume factor correction
            p = p * distance * distance
          else if (distance_prior.eq.2) then
            p = p * (distance * distance * exp(-distance / prior_parameter))
          endif
        endif
        model_params(m_count, prob+2) = p
        model_params(m_count, prob+3) = i
        if (debug) then
          write(66, *) models(i, model_column_count), models(i, abs_mag), &
                       models(i, model_columns), models(i, model_column_count-1), -99, &
                       model_params(m_count, 1:prob+2), mag - models(i, abs_mag), L_model, L_sed, p, i, distance_prior
        endif
        m_count = m_count + 1
      endif
    enddo
    m_count = m_count - 1
end subroutine find_best

end module model_fitter
