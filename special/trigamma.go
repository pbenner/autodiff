/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/trigamma.hpp
 */

//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

func trigamma_prec(x float64) float64 {
  // Max error in interpolated form: 3.736e-017
  const offset float64 = 2.1093254089355469
  P_1_2 := NewPolynomial([]float64{
    -1.1093280605946045,
    -3.8310674472619321,
    -3.3703848401898283,
     0.28080574467981213,
     1.6638069578676164,
     0.64468386819102836 })
  Q_1_2 := NewPolynomial([]float64{
     1.0,
     3.4535389668541151,
     4.5208926987851437,
     2.7012734178351534,
     0.64468798399785611,
    -0.20314516859987728e-6 })
  // Max error in interpolated form: 1.159e-017
  P_2_4 := NewPolynomial([]float64{
    -0.13803835004508849e-7,
     0.50000049158540261,
     1.6077979838469348,
     2.5645435828098254,
     2.0534873203680393,
     0.74566981111565923 })
  Q_2_4 := NewPolynomial([]float64{
     1.0,
     2.8822787662376169,
     4.1681660554090917,
     2.7853527819234466,
     0.74967671848044792,
    -0.00057069112416246805 })
  // Maximum Deviation Found:                     6.896e-018
  // Expected Error Term :                       -6.895e-018
  // Maximum Relative Change in Control Points :  8.497e-004
  P_4_inf := NewPolynomial([]float64{
    0.68947581948701249e-17,
    0.49999999999998975,
    1.0177274392923795,
    2.498208511343429,
    2.1921221359427595,
    1.5897035272532764,
    0.40154388356961734 })
  Q_4_inf := NewPolynomial([]float64{
    1.0,
    1.7021215452463932,
    4.4290431747556469,
    2.9745631894384922,
    2.3013614809773616,
    0.28360399799075752,
    0.022892987908906897 })

  if x <= 2.0 {
    return (offset + P_1_2.Eval(x)/Q_1_2.Eval(x))/(x*x)
  } else if x <= 4.0 {
    y := 1.0/x
    return (1.0 + P_2_4.Eval(y)/Q_2_4.Eval(y))/x
  }
  y := 1.0/x
  return (1.0 + P_4_inf.Eval(y)/Q_4_inf.Eval(y))/x
}

func trigamma_imp(x float64) float64 {
  result := 0.0
  //
  // Check for negative arguments and use reflection:
  //
  if x <= 0.0 {
    // Reflect:
    z := 1.0 - x;
    // Argument reduction for tan:
    if math.Floor(x) == x {
      return math.NaN()
    }
    s := 0.0
    if math.Abs(x) < math.Abs(z) {
      s = math.Sin(math.Pi*x)
    } else {
      s = math.Sin(math.Pi*z)
    }
    return -trigamma_imp(z) + math.Pow(math.Pi, 2)/(s*s)
  }
  if x < 1.0 {
    result = 1.0/(x*x)
    x     += 1.0
  }
  return result + trigamma_prec(x)
}

/* -------------------------------------------------------------------------- */

func Trigamma(x float64) float64 {
  return trigamma_imp(x)
}
