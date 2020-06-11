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

// Element-wise addition of two matrices.
func MaddM(a, b Matrix) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MaddM(a, b)
	return r
}

// Add scalar b to all elements of a.
func MaddS(a Matrix, b Scalar) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MaddS(a, b)
	return r
}

// Element-wise substraction of two matrices.
func MsubM(a, b Matrix) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MsubM(a, b)
	return r
}

// Substract b from all elements of a.
func MsubS(a Matrix, b Scalar) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MsubS(a, b)
	return r
}

// Element-wise multiplication of two matrices.
func MmulM(a, b Matrix) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MmulM(a, b)
	return r
}

// Multiply all elements of a with b.
func MmulS(a Matrix, b Scalar) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MmulS(a, b)
	return r
}

// Element-wise division of two matrices.
func MdivM(a, b Matrix) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MdivM(a, b)
	return r
}

// Divide all elements of a by b.
func MdivS(a Matrix, b Scalar) Matrix {
	n, m := a.Dims()
	r := NullMatrix(a.ElementType(), n, m)
	r.MdivS(a, b)
	return r
}

// Matrix product of a and b.
func MdotM(a, b Matrix) Matrix {
	n1, _ := a.Dims()
	_, m2 := b.Dims()
	r := NullMatrix(a.ElementType(), n1, m2)
	r.MdotM(a, b)
	return r
}

// Matrix vector product of a and b.
func MdotV(a Matrix, b Vector) Vector {
	n, _ := a.Dims()
	r := NullVector(a.ElementType(), n)
	r.MdotV(a, b)
	return r
}

// Vector matrix product of a and b.
func VdotM(a Vector, b Matrix) Vector {
	_, m := b.Dims()
	r := NullVector(a.ElementType(), m)
	r.VdotM(a, b)
	return r
}

// Outer product of two vectors.
func Outer(a, b Vector) Matrix {
	r := NullMatrix(a.ElementType(), a.Dim(), b.Dim())
	r.Outer(a, b)
	return r
}

// Returns the trace of a.
func Mtrace(a Matrix) Scalar {
	r := a.At(0, 0).CloneScalar()
	r.Mtrace(a)
	return r
}

// Frobenius norm.
func Mnorm(a Matrix) Scalar {
	n, m := a.Dims()
	if n == 0 || m == 0 {
		return nil
	}
	c := NewBareReal(2.0)
	t := NewScalar(a.ElementType(), 0.0)
	s := NewScalar(a.ElementType(), 0.0)
	v := a.AsVector()
	s.Pow(v.At(0), NewBareReal(2.0))
	for i := 1; i < v.Dim(); i++ {
		t.Pow(v.At(i), c)
		s.Add(s, t)
	}
	return s
}
