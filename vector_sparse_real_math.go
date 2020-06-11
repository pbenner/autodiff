/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2019 Philipp Benner
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
/* -------------------------------------------------------------------------- */
package autodiff

/* -------------------------------------------------------------------------- */
import "math"

/* -------------------------------------------------------------------------- */
// Test if elements in a equal elements in b.
func (a *SparseRealVector) Equals(b ConstVector, epsilon float64) bool {
	if a.Dim() != b.Dim() {
		panic("VEqual(): Vector dimensions do not match!")
	}
	for it := a.JOINT_ITERATOR(b); it.Ok(); it.Next() {
		s1, s2 := it.GET()
		if s1 == nil {
			return false
		}
		if !s1.Equals(s2, epsilon) {
			return false
		}
	}
	return true
}
func (a *SparseRealVector) EQUALS(b *SparseRealVector, epsilon float64) bool {
	if a.Dim() != b.Dim() {
		panic("VEqual(): Vector dimensions do not match!")
	}
	for it := a.JOINT_ITERATOR_(b); it.Ok(); it.Next() {
		s1, s2 := it.GET()
		if s1 == nil {
			return false
		}
		if s2 == nil {
			return false
		}
		if !s1.EQUALS(s2, epsilon) {
			return false
		}
	}
	return true
}

/* -------------------------------------------------------------------------- */
// Element-wise addition of two vectors. The result is stored in r.
func (r *SparseRealVector) VaddV(a, b ConstVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		s_r.Add(s_a, s_b)
	}
	return r
}
func (r *SparseRealVector) VADDV(a, b *SparseRealVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		switch {
		case s_a == nil && s_b == nil:
			s_r.SetValue(0.0)
		case s_b == nil:
			s_r.SET(s_a)
		case s_a == nil:
			s_r.SET(s_b)
		default:
			s_r.ADD(s_a, s_b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VaddS(a ConstVector, b ConstScalar) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		s_r.Add(s_a, b)
	}
	return r
}
func (r *SparseRealVector) VADDS(a *SparseRealVector, b *Real) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		if s_a == nil {
			s_r.SET(b)
		} else {
			s_r.ADD(s_a, b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substraction of two vectors. The result is stored in r.
func (r *SparseRealVector) VsubV(a, b ConstVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		s_r.Sub(s_a, s_b)
	}
	return r
}
func (r *SparseRealVector) VSUBV(a, b *SparseRealVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		switch {
		case s_a == nil && s_b == nil:
			s_r.SetValue(0.0)
		case s_b == nil:
			s_r.SET(s_a)
		case s_a == nil:
			s_r.SET(s_b)
			s_r.NEG(s_r)
		default:
			s_r.SUB(s_a, s_b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substractor of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VsubS(a ConstVector, b ConstScalar) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		s_r.Sub(s_a, b)
	}
	return r
}
func (r *SparseRealVector) VSUBS(a *SparseRealVector, b *Real) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		if s_a == nil {
			s_r.SET(b)
			s_r.NEG(s_r)
		} else {
			s_r.SUB(s_a, b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two vectors. The result is stored in r.
func (r *SparseRealVector) VmulV(a, b ConstVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			continue
		}
		s_r.Mul(s_a, s_b)
	}
	return r
}
func (r *SparseRealVector) VMULV(a, b *SparseRealVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		switch {
		case s_a == nil || s_b == nil:
			s_r.SetValue(0.0)
		default:
			s_r.MUL(s_a, s_b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VmulS(a ConstVector, s ConstScalar) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			continue
		}
		s_r.Mul(s_a, s)
	}
	return r
}
func (r *SparseRealVector) VMULS(a *SparseRealVector, b *Real) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		if s_a == nil {
			s_r.SetValue(0.0)
		} else {
			s_r.MUL(s_a, b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise division of two vectors. The result is stored in r.
func (r *SparseRealVector) VdivV(a, b ConstVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			continue
		}
		s_r.Div(s_a, s_b)
	}
	return r
}
func (r *SparseRealVector) VDIVV(a, b *SparseRealVector) Vector {
	if n := r.Dim(); a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		s_b := it.s3
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		switch {
		case s_b == nil:
			s_r.SetValue(math.NaN())
		case s_a == nil:
			s_r.SetValue(0.0)
		default:
			s_r.MUL(s_a, s_b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise division of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VdivS(a ConstVector, s ConstScalar) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			continue
		}
		s_r.Div(s_a, s)
	}
	return r
}
func (r *SparseRealVector) VDIVS(a *SparseRealVector, b *Real) Vector {
	if r.Dim() != a.Dim() {
		panic("vector dimensions do not match")
	}
	for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
		s_r := it.s1
		s_a := it.s2
		if s_r == nil {
			s_r = r.AT(it.Index())
		}
		if s_a == nil {
			s_r.SetValue(0.0)
		} else {
			s_r.DIV(s_a, b)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Matrix vector product of a and b. The result is stored in r.
func (r *SparseRealVector) MdotV(a ConstMatrix, b ConstVector) Vector {
	n, m := a.Dims()
	if r.Dim() != n || b.Dim() != m {
		panic("matrix/vector dimensions do not match!")
	}
	if n == 0 || m == 0 {
		return r
	}
	if r.AT(0) == b.ConstAt(0) {
		panic("result and argument must be different vectors")
	}
	t := NullReal()
	for i := 0; i < n; i++ {
		r.AT(i).Reset()
		for j := 0; j < m; j++ {
			t.Mul(a.ConstAt(i, j), b.ConstAt(j))
			r.AT(i).ADD(r.AT(i), t)
		}
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Vector matrix product of a and b. The result is stored in r.
func (r *SparseRealVector) VdotM(a ConstVector, b ConstMatrix) Vector {
	n, m := b.Dims()
	if r.Dim() != m || a.Dim() != n {
		panic("matrix/vector dimensions do not match!")
	}
	if n == 0 || m == 0 {
		return r
	}
	if r.AT(0) == a.ConstAt(0) {
		panic("result and argument must be different vectors")
	}
	t := NullReal()
	for i := 0; i < m; i++ {
		r.AT(i).Reset()
		for j := 0; j < n; j++ {
			t.Mul(a.ConstAt(j), b.ConstAt(j, i))
			r.AT(i).ADD(r.AT(i), t)
		}
	}
	return r
}
