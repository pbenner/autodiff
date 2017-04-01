/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/factorial.hpp
 */

//  Copyright John Maddock 2006, 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

var factorialList = []int64{
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6402373705728000,
  121645100408832000,
  0.243290200817664e19 }

var factorialMax = len(factorialList)

func Factorial(x int) float64 {
  if x < factorialMax {
    return float64(factorialList[x])
  }
  return math.Floor(math.Gamma(float64(x+1)) + 0.5)
}
