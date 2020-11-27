module unidam_extra_functions
contains
real*8 function student_pdf_single ( x, a, b, c)

!*****************************************************************************80
!
!! STUDENT_PDF evaluates the central Student T PDF.
!
!  Discussion:
!
!    PDF(A,B,C;X) = Gamma ( (C+1)/2 ) /
!      ( Gamma ( C / 2 ) * Sqrt ( PI * C )
!      * ( 1 + ((X-A)/B)^2/C )^(C + 1/2 ) )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    02 November 2005
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the PDF.
!
!    Input, real ( kind = 8 ) A, B, shape parameters of the PDF,
!    used to transform the argument X to a shifted and scaled
!    value Y = ( X - A ) / B.  It is required that B be nonzero.
!    For the standard distribution, A = 0 and B = 1.
!
!    Input, real ( kind = 8 ) C, is usually called the number of
!    degrees of freedom of the distribution.  C is typically an
!    integer, but that is not essential.  It is required that
!    C be strictly positive.
!
!    Output, real ( kind = 8 ) PDF, the value of the PDF.
!
  implicit none

  real ( kind = 8 ) a
  real ( kind = 8 ) b
  real ( kind = 8 ) c
  !real ( kind = 8 ) pdf
  real ( kind = 8 ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = 8 ) x
  real ( kind = 8 ) y

  y = ( x - a ) / b

  student_pdf_single = gamma ( 0.5D+00 * ( c + 1.0D+00 ) ) / ( sqrt ( r8_pi * c ) &
    * gamma ( 0.5D+00 * c ) &
    * sqrt ( ( 1.0D+00 + y * y / c ) ** ( c + 1.0D+00 ) ) )
end

subroutine student_pdf ( n, x, a, b, c, y)

!*****************************************************************************80
!
!! STUDENT_PDF evaluates the central Student T PDF.
!
!  Discussion:
!
!    PDF(A,B,C;X) = Gamma ( (C+1)/2 ) /
!      ( Gamma ( C / 2 ) * Sqrt ( PI * C )
!      * ( 1 + ((X-A)/B)^2/C )^(C + 1/2 ) )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    02 November 2005
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the PDF.
!
!    Input, real ( kind = 8 ) A, B, shape parameters of the PDF,
!    used to transform the argument X to a shifted and scaled
!    value Y = ( X - A ) / B.  It is required that B be nonzero.
!    For the standard distribution, A = 0 and B = 1.
!
!    Input, real ( kind = 8 ) C, is usually called the number of
!    degrees of freedom of the distribution.  C is typically an
!    integer, but that is not essential.  It is required that
!    C be strictly positive.
!
!    Output, real ( kind = 8 ) PDF, the value of the PDF.
!
  implicit none

  real ( kind = 8 ) a
  real ( kind = 8 ) b
  real ( kind = 8 ) c
  !real ( kind = 8 ) pdf
  real ( kind = 8 ), parameter :: r8_pi = 3.141592653589793D+00
  integer, intent(in) :: n
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xx(n)
  real ( kind = 8 ), intent(out) :: y(n)

  xx = ( x - a ) / b

  y = exp( log_gamma( 0.5D+00 * ( c + 1.0D+00 ) ) - 0.5 * log( r8_pi * c ) &
      - log_gamma(0.5 * c) - ( 0.5 * c + 0.5D+00 ) * log( 1.0D+00 + xx * xx / c ))
  !y = gamma ( 0.5D+00 * ( c + 1.0D+00 ) ) / ( sqrt ( r8_pi * c ) &
  !  * gamma ( 0.5D+00 * c ) &
  !  * ( 1.0D+00 + xx * xx / c ) ** ( 0.5 * c + 0.5D+00 ) ) 
end

subroutine normal_01_cdf ( x, cdf )

!*****************************************************************************80
!
!! NORMAL_01_CDF evaluates the Normal 01 CDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    AG Adams,
!    Algorithm 39,
!    Areas Under the Normal Curve,
!    Computer Journal,
!    Volume 12, pages 197-198, 1969.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the CDF.
!
!    Output, real ( kind = 8 ) CDF, the value of the CDF.
!
  implicit none

  real ( kind = 8 ), parameter :: a1 = 0.398942280444D+00
  real ( kind = 8 ), parameter :: a2 = 0.399903438504D+00
  real ( kind = 8 ), parameter :: a3 = 5.75885480458D+00
  real ( kind = 8 ), parameter :: a4 = 29.8213557808D+00
  real ( kind = 8 ), parameter :: a5 = 2.62433121679D+00
  real ( kind = 8 ), parameter :: a6 = 48.6959930692D+00
  real ( kind = 8 ), parameter :: a7 = 5.92885724438D+00
  real ( kind = 8 ), parameter :: b0 = 0.398942280385D+00
  real ( kind = 8 ), parameter :: b1 = 3.8052D-08
  real ( kind = 8 ), parameter :: b2 = 1.00000615302D+00
  real ( kind = 8 ), parameter :: b3 = 3.98064794D-04
  real ( kind = 8 ), parameter :: b4 = 1.98615381364D+00
  real ( kind = 8 ), parameter :: b5 = 0.151679116635D+00
  real ( kind = 8 ), parameter :: b6 = 5.29330324926D+00
  real ( kind = 8 ), parameter :: b7 = 4.8385912808D+00
  real ( kind = 8 ), parameter :: b8 = 15.1508972451D+00
  real ( kind = 8 ), parameter :: b9 = 0.742380924027D+00
  real ( kind = 8 ), parameter :: b10 = 30.789933034D+00
  real ( kind = 8 ), parameter :: b11 = 3.99019417011D+00
  real ( kind = 8 ) cdf
  real ( kind = 8 ) q
  real ( kind = 8 ) x
  real ( kind = 8 ) y
!
!  |X| <= 1.28.
!
  if ( abs ( x ) <= 1.28D+00 ) then

    y = 0.5D+00 * x * x

    q = 0.5D+00 - abs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5 &
      + a6 / ( y + a7 ) ) ) )
!
!  1.28 < |X| <= 12.7
!
  else if ( abs ( x ) <= 12.7D+00 ) then

    y = 0.5D+00 * x * x

    q = exp ( - y ) * b0 / ( abs ( x ) - b1 &
      + b2 / ( abs ( x ) + b3 &
      + b4 / ( abs ( x ) - b5 &
      + b6 / ( abs ( x ) + b7 &
      - b8 / ( abs ( x ) + b9 &
      + b10 / ( abs ( x ) + b11 ) ) ) ) ) )
!
!  12.7 < |X|
!
  else

    q = 0.0D+00

  end if
!
!  Take account of negative X.
!
  if ( x < 0.0D+00 ) then
    cdf = q
  else
    cdf = 1.0D+00 - q
  end if

  return
end

real*8 function normal_cdf ( x )

!*****************************************************************************80
!
!! NORMAL_01_CDF evaluates the Normal 01 CDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 February 1999
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    AG Adams,
!    Algorithm 39,
!    Areas Under the Normal Curve,
!    Computer Journal,
!    Volume 12, pages 197-198, 1969.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the CDF.
!
!    Output, real ( kind = 8 ) CDF, the value of the CDF.
!
  implicit none

  real ( kind = 8 ), parameter :: a1 = 0.398942280444D+00
  real ( kind = 8 ), parameter :: a2 = 0.399903438504D+00
  real ( kind = 8 ), parameter :: a3 = 5.75885480458D+00
  real ( kind = 8 ), parameter :: a4 = 29.8213557808D+00
  real ( kind = 8 ), parameter :: a5 = 2.62433121679D+00
  real ( kind = 8 ), parameter :: a6 = 48.6959930692D+00
  real ( kind = 8 ), parameter :: a7 = 5.92885724438D+00
  real ( kind = 8 ), parameter :: b0 = 0.398942280385D+00
  real ( kind = 8 ), parameter :: b1 = 3.8052D-08
  real ( kind = 8 ), parameter :: b2 = 1.00000615302D+00
  real ( kind = 8 ), parameter :: b3 = 3.98064794D-04
  real ( kind = 8 ), parameter :: b4 = 1.98615381364D+00
  real ( kind = 8 ), parameter :: b5 = 0.151679116635D+00
  real ( kind = 8 ), parameter :: b6 = 5.29330324926D+00
  real ( kind = 8 ), parameter :: b7 = 4.8385912808D+00
  real ( kind = 8 ), parameter :: b8 = 15.1508972451D+00
  real ( kind = 8 ), parameter :: b9 = 0.742380924027D+00
  real ( kind = 8 ), parameter :: b10 = 30.789933034D+00
  real ( kind = 8 ), parameter :: b11 = 3.99019417011D+00
  real ( kind = 8 ) q
  real ( kind = 8 ) x
  real ( kind = 8 ) y
!
!  |X| <= 1.28.
!
  if ( abs ( x ) <= 1.28D+00 ) then

    y = 0.5D+00 * x * x

    q = 0.5D+00 - abs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5 &
      + a6 / ( y + a7 ) ) ) )
!
!  1.28 < |X| <= 12.7
!
  else if ( abs ( x ) <= 12.7D+00 ) then

    y = 0.5D+00 * x * x

    q = exp ( - y ) * b0 / ( abs ( x ) - b1 &
      + b2 / ( abs ( x ) + b3 &
      + b4 / ( abs ( x ) - b5 &
      + b6 / ( abs ( x ) + b7 &
      - b8 / ( abs ( x ) + b9 &
      + b10 / ( abs ( x ) + b11 ) ) ) ) ) )
!
!  12.7 < |X|
!
  else

    q = 0.0D+00

  end if
!
!  Take account of negative X.
!
  if ( x < 0.0D+00 ) then
    normal_cdf = q
  else
    normal_cdf = 1.0D+00 - q
  end if

  return
end

subroutine normal_01_pdf ( n, x, pdf )

!*****************************************************************************80
!
!! NORMAL_01_PDF evaluates the Normal 01 PDF.
!
!  Discussion:
!
!    The Normal 01 PDF is also called the "Standard Normal" PDF, or
!    the Normal PDF with 0 mean and variance 1.
!
!    PDF(X) = exp ( - 0.5 * X^2 ) / sqrt ( 2 * PI )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    04 December 1999
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the PDF.
!
!    Output, real ( kind = 8 ) PDF, the value of the PDF.
!
  implicit none

  integer n
  real ( kind = 8 ) pdf(n)
  real ( kind = 8 ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = 8 ) x(n)

  pdf = exp ( -0.5D+00 * x * x ) / sqrt ( 2.0D+00 * r8_pi )

  return
end

subroutine normal_truncated_ab_pdf ( n, x, mu, s, a, b, pdf )

!*****************************************************************************80
!
!! NORMAL_TRUNCATED_AB_PDF evaluates the truncated Normal PDF.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    14 August 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the PDF.
!
!    Input, real ( kind = 8 ) MU, S, the mean and standard deviation of the
!    parent Normal distribution.
!
!    Input, real ( kind = 8 ) A, B, the lower and upper truncation limits.
!
!    Output, real ( kind = 8 ) PDF, the value of the PDF.
!
  implicit none

  integer n
  real ( kind = 8 ) a
  real ( kind = 8 ) alpha
  real ( kind = 8 ) alpha_cdf
  real ( kind = 8 ) b
  real ( kind = 8 ) beta
  real ( kind = 8 ) beta_cdf
  real ( kind = 8 ) mu
  real ( kind = 8 ) pdf(n)
  real ( kind = 8 ) s
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xi(n)
  real ( kind = 8 ) xi_pdf(n)

  alpha = ( a - mu ) / s
  beta = ( b - mu ) / s
  xi = ( x - mu ) / s

  !call normal_01_cdf ( 1, alpha, alpha_cdf )
  !call normal_01_cdf ( 1, beta, beta_cdf )
  alpha_cdf=1
  beta_cdf = 1
  call normal_01_pdf ( n, xi, xi_pdf )

  pdf = xi_pdf / ( beta_cdf - alpha_cdf ) / s

  return
end

subroutine norm_cdf(x, res)
  real*8, intent(in) :: x
  real*8, intent(out) :: res
  res =  0.5 * (1 + erf(x / sqrt(2.0)))
end subroutine norm_cdf

subroutine norm_cdf_arr(n, x, res)
integer n
  real*8, intent(in) :: x(n)
  real*8, intent(out) :: res(n)
  res =  0.5 * (1 + erf(x / sqrt(2.0)))
end subroutine norm_cdf_arr

subroutine trunc_normal(n, x, mu, s, a, b, res)
  integer n
  real*8, intent(in) :: x(n)
  real*8, intent(in) :: mu, s, a, b
  real*8, intent(out) :: res(n)
  real ( kind = 8 ) alpha
  real ( kind = 8 ) alpha_cdf
  real ( kind = 8 ) beta
  real ( kind = 8 ) beta_cdf
  real ( kind = 8 ) xi(n)
  real ( kind = 8 ) xi_pdf(n)

  call norm_cdf(( a - mu ) / s, alpha_cdf)
  call norm_cdf(( b - mu ) / s, beta_cdf)
  xi = ( x - mu ) / s
  call normal_01_pdf ( n, xi, xi_pdf )
  res = xi_pdf / ( beta_cdf - alpha_cdf ) / s

end subroutine trunc_normal

subroutine skew_normal_pdf( x, a, b, c, pdf)
!
!  The purpose of this subroutine is to return the probability distribution
!  function (pdf) value for a given x-axis location, given the parameters for
!  the skew-normal distribution.  It is assumed that the parameters are 
!  acceptable, the subroutine skew_normal_check should be called before.
!
!  Author: Kenny Anderson
!  Created: January 2012
!
implicit none
real*8, intent(in) :: x
real*8, intent(in) :: a
real*8, intent(in) :: b
real*8, intent(in) :: c
real*8, intent(out) :: pdf
real*8, parameter :: pi = &
  3.14159265358979323846264338327950288419716939937510E+00
real*8 r1
call norm_cdf(c*(x-a)/b, r1)
pdf = (1/(b*sqrt(0.5 * pi))*(exp(-(x-a)**2/(2*b**2))))*r1
return
end subroutine skew_normal_pdf


subroutine skew_normal_pdf_arr( n, x, a, b, c, pdf)
!
!  The purpose of this subroutine is to return the probability distribution
!  function (pdf) value for a given x-axis location, given the parameters for
!  the skew-normal distribution.  It is assumed that the parameters are 
!  acceptable, the subroutine skew_normal_check should be called before.
!
!  Author: Kenny Anderson
!  Created: January 2012
!
implicit none
integer n
real*8, intent(in) :: x(n)
real*8, intent(in) :: a
real*8, intent(in) :: b
real*8, intent(in) :: c
real*8, intent(out) :: pdf(n)
real*8, parameter :: pi = &
  3.14159265358979323846264338327950288419716939937510E+00
real*8 r1(n)
call norm_cdf_arr(n, c*(x-a)/b, r1)
pdf = (1/(b*sqrt(0.5 * pi))*(exp(-(x-a)**2/(2*b**2))))*r1
return
end subroutine skew_normal_pdf_arr
end module unidam_extra_functions
