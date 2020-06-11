/* Copyright (C) 2015, 2016, 2017 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package simple

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

// Element-wise addition of two vectors.
func VaddV(a, b Vector) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VaddV(a, b)
	return r
}

// Element-wise addition of a vector and a scalar.
func VaddS(a Vector, b Scalar) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VaddS(a, b)
	return r
}

// Element-wise substraction of two vectors.
func VsubV(a, b Vector) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VsubV(a, b)
	return r
}

// Element-wise substractor of a vector and a scalar.
func VsubS(a Vector, b Scalar) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VsubS(a, b)
	return r
}

// Element-wise multiplication of two vectors.
func VmulV(a Vector, b Vector) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VmulV(a, b)
	return r
}

// Element-wise substraction of a vector and a scalar.
func VmulS(a Vector, s Scalar) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VmulS(a, s)
	return r
}

// Element-wise division of two vectors.
func VdivV(a Vector, b Vector) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VdivV(a, b)
	return r
}

// Element-wise division of a vector and a scalar.
func VdivS(a Vector, s Scalar) Vector {
	r := NullVector(a.ElementType(), a.Dim())
	r.VdivS(a, s)
	return r
}

func VdotV(a, b Vector) Scalar {
	r := NullScalar(a.ElementType())
	r.VdotV(a, b)
	return r
}
