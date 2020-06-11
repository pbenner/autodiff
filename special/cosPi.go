/* Copyright (C) 2017 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/cos_pi.hpp
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

func CosPi(x float64) float64 {
	// cos of pi*x:
	invert := false
	if math.Abs(x) < 0.25 {
		return math.Cos(math.Pi * x)
	}
	if x < 0 {
		x = -x
	}

	rem := math.Floor(x)
	if int(rem)&1 != 0 {
		invert = !invert
	}
	rem = x - rem
	if rem > 0.5 {
		rem = 1 - rem
		invert = !invert
	}
	if rem == 0.5 {
		return 0
	}
	if rem > 0.25 {
		rem = 0.5 - rem
		rem = math.Sin(math.Pi * rem)
	} else {
		rem = math.Cos(math.Pi * rem)
	}
	if invert {
		return -rem
	} else {
		return rem
	}
}
