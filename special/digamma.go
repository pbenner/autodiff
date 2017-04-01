/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/digamma.hpp
 */

//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

const digamma_large_lim float64 = 10

/* -------------------------------------------------------------------------- */

func digamma_imp_large(x float64) float64 {
  P := NewPolynomial([]float64{
     0.083333333333333333333333333333333333333333333333333,
    -0.0083333333333333333333333333333333333333333333333333,
     0.003968253968253968253968253968253968253968253968254,
    -0.0041666666666666666666666666666666666666666666666667,
     0.0075757575757575757575757575757575757575757575757576,
    -0.021092796092796092796092796092796092796092796092796,
     0.083333333333333333333333333333333333333333333333333,
    -0.44325980392156862745098039215686274509803921568627 })
  x -= 1.0
  z := 1.0/(x*x)
  result := math.Log(x)
  result += 1.0/(2.0*x)
  result -= z * P.Eval(z)
  return result
}

func digamma_imp_1_2(x float64) float64 {
  //
  // Now the approximation, we use the form:
  //
  // digamma(x) = (x - root) * (Y + R(x-1))
  //
  // Where root is the location of the positive root of digamma,
  // Y is a constant, and R is optimised for low absolute error
  // compared to Y.
  //
  // Maximum Deviation Found:               1.466e-18
  // At double precision, max error found:  2.452e-17
  //
  Y := 0.99558162689208984

  root1 := 1569415565.0 / 1073741824.0
  root2 := (381566830.0 / 1073741824.0) / 1073741824.0
  root3 := 0.9016312093258695918615325266959189453125e-19

  P := NewPolynomial([]float64{
     0.25479851061131551,
    -0.32555031186804491,
    -0.65031853770896507,
    -0.28919126444774784,
    -0.045251321448739056,
    -0.0020713321167745952 })
  Q := NewPolynomial([]float64{
     1.0,
     2.0767117023730469,
     1.4606242909763515,
     0.43593529692665969,
     0.054151797245674225,
     0.0021284987017821144,
    -0.55789841321675513e-6 })
  g := x - root1
  g -= root2
  g -= root3
  r := P.Eval(x-1.0)/Q.Eval(x-1.0)

  return g * Y + g * r
}

func digamma_imp(x float64) float64 {
  result := 0.0
  //
  // Check for negative arguments and use reflection:
  //
  if x <= -1.0 {
    // Reflect:
    x = 1.0 - x
    // Argument reduction for tan:
    remainder := x - math.Floor(x)
    // Shift to negative if > 0.5:
    if remainder > 0.5 {
      remainder -= 1.0
    }
    //
    // check for evaluation at a negative pole:
    //
    if remainder == 0 {
      return math.NaN()
    }
    result = math.Pi / math.Tan(math.Pi*remainder)
  }
  if x == 0 {
    return math.NaN()
  }
  //
  // If we're above the lower-limit for the
  // asymptotic expansion then use it:
  //
  if x >= digamma_large_lim {
    result += digamma_imp_large(x)
  } else {
    //
    // If x > 2 reduce to the interval [1,2]:
    //
    for x > 2.0 {
      x      -= 1.0
      result += 1.0/x
    }
    //
    // If x < 1 use recurrance to shift to > 1:
    //
    for x < 1.0 {
      result -= 1.0/x
      x      += 1.0
    }
    result += digamma_imp_1_2(x)
  }
  return result
}

/* -------------------------------------------------------------------------- */

func Digamma(x float64) float64 {
  return digamma_imp(x)
}
