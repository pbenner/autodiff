/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2017 Philipp Benner
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
// Test if elements in a equal elements in b.
func (a DenseRealVector) Equals(b ConstVector, epsilon float64) bool {
	if a.Dim() != b.Dim() {
		panic("VEqual(): Vector dimensions do not match!")
	}
	for i := 0; i < a.Dim(); i++ {
		if !a.ConstAt(i).Equals(b.ConstAt(i), epsilon) {
			return false
		}
	}
	return true
}
func (a DenseRealVector) EQUALS(b DenseRealVector, epsilon float64) bool {
	if a.Dim() != b.Dim() {
		panic("VEqual(): Vector dimensions do not match!")
	}
	for i := 0; i < a.Dim(); i++ {
		if !a.AT(i).EQUALS(b.AT(i), epsilon) {
			return false
		}
	}
	return true
}

/* -------------------------------------------------------------------------- */
// Element-wise addition of two vectors. The result is stored in r.
func (r DenseRealVector) VaddV(a, b ConstVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Add(a.ConstAt(i), b.ConstAt(i))
	}
	return r
}
func (r DenseRealVector) VADDV(a, b DenseRealVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).ADD(a.AT(i), b.AT(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r DenseRealVector) VaddS(a ConstVector, b ConstScalar) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Add(a.ConstAt(i), b)
	}
	return r
}
func (r DenseRealVector) VADDS(a DenseRealVector, b *Real) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).ADD(a.AT(i), b)
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substraction of two vectors. The result is stored in r.
func (r DenseRealVector) VsubV(a, b ConstVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Sub(a.ConstAt(i), b.ConstAt(i))
	}
	return r
}
func (r DenseRealVector) VSUBV(a, b DenseRealVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).SUB(a.AT(i), b.AT(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substractor of a vector and a scalar. The result is stored in r.
func (r DenseRealVector) VsubS(a ConstVector, b ConstScalar) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Sub(a.ConstAt(i), b)
	}
	return r
}
func (r DenseRealVector) VSUBS(a DenseRealVector, b *Real) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).SUB(a.AT(i), b)
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two vectors. The result is stored in r.
func (r DenseRealVector) VmulV(a, b ConstVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Mul(a.ConstAt(i), b.ConstAt(i))
	}
	return r
}
func (r DenseRealVector) VMULV(a, b DenseRealVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).MUL(a.AT(i), b.AT(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r DenseRealVector) VmulS(a ConstVector, s ConstScalar) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Mul(a.ConstAt(i), s)
	}
	return r
}
func (r DenseRealVector) VMULS(a DenseRealVector, s *Real) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).MUL(a.AT(i), s)
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise division of two vectors. The result is stored in r.
func (r DenseRealVector) VdivV(a, b ConstVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Div(a.ConstAt(i), b.ConstAt(i))
	}
	return r
}
func (r DenseRealVector) VDIVV(a, b DenseRealVector) Vector {
	n := r.Dim()
	if a.Dim() != n || b.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).DIV(a.AT(i), b.AT(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Element-wise division of a vector and a scalar. The result is stored in r.
func (r DenseRealVector) VdivS(a ConstVector, s ConstScalar) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).Div(a.ConstAt(i), s)
	}
	return r
}
func (r DenseRealVector) VDIVS(a DenseRealVector, s *Real) Vector {
	n := r.Dim()
	if a.Dim() != n {
		panic("vector dimensions do not match")
	}
	for i := 0; i < a.Dim(); i++ {
		r.AT(i).DIV(a.AT(i), s)
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Matrix vector product of a and b. The result is stored in r.
func (r DenseRealVector) MdotV(a ConstMatrix, b ConstVector) Vector {
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
func (r DenseRealVector) VdotM(a ConstVector, b ConstMatrix) Vector {
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
