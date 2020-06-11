/* Copyright (C) 2017 Philipp Benner
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

package givensRotation

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func apply(a1, a2, c, s Scalar, t1, t2 Scalar) {
	t1.Set(a1)
	// update a1
	a1.Mul(c, a1)
	t2.Mul(s, a2)
	a1.Sub(a1, t2)
	// update a2
	a2.Mul(c, a2)
	t2.Mul(s, t1)
	a2.Add(a2, t2)
}

/* -------------------------------------------------------------------------- */

func ApplyBidiagLeft(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	_, n := A.Dims()
	{
		j := i
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	{
		j := k
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i + 1; j < n && j != k {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k + 1; j < n && j != i {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
}

func ApplyBidiagRight(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	{
		j := i
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	{
		j := k
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i - 1; j >= 0 && j != k {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k - 1; j >= 0 && j != i {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
}

/* -------------------------------------------------------------------------- */

func ApplyTridiagLeft(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	_, n := A.Dims()
	{
		j := i
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	{
		j := k
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i + 1; j < n && j != k && j != k-1 {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i - 1; j >= 0 && j != k {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k + 1; j < n && j != i && j != i-1 {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k - 1; j >= 0 && j != i {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
}

func ApplyTridiagRight(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	m, _ := A.Dims()
	{
		j := i
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	{
		j := k
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i - 1; j >= 0 && j != k && j != k+1 {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := i + 1; j < m && j != k {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k - 1; j >= 0 && j != i && j != i+1 {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
	if j := k + 1; j < m && j != i {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
}

/* -------------------------------------------------------------------------- */

func ApplyHessenbergLeft(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	_, n := A.Dims()

	var r int
	if i < k {
		r = i - 1
	} else {
		r = k - 1
	}
	if r < 0 {
		r = 0
	}
	for j := r; j < n; j++ {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
}

func ApplyHessenbergRight(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	m, _ := A.Dims()

	var r int
	if i > k {
		r = i + 2
	} else {
		r = k + 2
	}
	if r > m {
		r = m
	}
	for j := 0; j < r; j++ {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
}

/* -------------------------------------------------------------------------- */

func ApplyLeft(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	_, n := A.Dims()

	for j := 0; j < n; j++ {
		a1 := A.At(i, j)
		a2 := A.At(k, j)
		apply(a1, a2, c, s, t1, t2)
	}
}

func ApplyRight(A Matrix, c, s Scalar, i, k int, t1, t2 Scalar) {
	m, _ := A.Dims()

	for j := 0; j < m; j++ {
		a1 := A.At(j, i)
		a2 := A.At(j, k)
		apply(a1, a2, c, s, t1, t2)
	}
}

/* -------------------------------------------------------------------------- */

// Compute c and s such that
// [  c  s ]^T  [ a ]  =  [ r ]
// [ -s  c ]    [ b ]  =  [ 0 ]
//
// where c = cos(theta)
//       s = sin(theta)
//
// => c =  a / sqrt(a^2 + b^2)
//    s = -b / sqrt(a^2 + b^2)

func Run(a, b, c, s Scalar) {
	c1 := BareReal(1.0)

	if b.GetValue() == 0.0 {
		c.SetValue(1.0)
		s.SetValue(0.0)
	} else {
		if math.Abs(b.GetValue()) > math.Abs(a.GetValue()) {
			c.Div(a, b)
			c.Neg(c)

			s.Mul(c, c)
			s.Add(s, &c1)
			s.Sqrt(s)
			s.Div(&c1, s)

			c.Mul(c, s)
		} else {
			s.Div(b, a)
			s.Neg(s)

			c.Mul(s, s)
			c.Add(c, &c1)
			c.Sqrt(c)
			c.Div(&c1, c)

			s.Mul(s, c)
		}
	}
}
