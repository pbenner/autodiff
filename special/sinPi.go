/* Copyright (C) 2017 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/sin_pi.hpp
 */

//  Copyright (c) 2007 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

func SinPi(x float64) float64 {

	if x < 0 {
		return -SinPi(-x)
	}
	// sin of pi*x:
	invert := false

	if x < 0.5 {
		return math.Sin(math.Pi * x)
	}
	if x < 1 {
		invert = true
		x = -x
	}

	rem := math.Floor(x)
	if int(rem)&1 != 0 {
		invert = !invert
	}
	rem = x - rem
	if rem > 0.5 {
		rem = 1 - rem
	}
	if rem == 0.5 {
		if invert {
			return -1
		} else {
			return 1
		}
	}
	rem = math.Sin(math.Pi * rem)

	if invert {
		return -rem
	} else {
		return rem
	}
}
