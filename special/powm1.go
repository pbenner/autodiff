/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/powm1.hpp
 */

//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

func Powm1(a, z float64) float64 {

  if math.Abs(a) < 1.0 || math.Abs(z) < 1 {
    p := math.Log(a) * z
    if math.Abs(p) < 2.0 {
      return math.Expm1(p)
    }
    // otherwise fall though:
  }
  return math.Pow(a, z) - 1.0
}
