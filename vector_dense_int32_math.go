/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2020 Philipp Benner
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
/* -------------------------------------------------------------------------- */
package autodiff
/* -------------------------------------------------------------------------- */
// Test if elements in a equal elements in b.
func (a DenseInt32Vector) Equals(b ConstVector, epsilon float64) bool {
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
func (a DenseInt32Vector) EQUALS(b DenseInt32Vector, epsilon float64) bool {
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
func (r DenseInt32Vector) VaddV(a, b ConstVector) Vector {
  n := r.Dim()
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Add(a.ConstAt(i), b.ConstAt(i))
  }
  return r
}
func (r DenseInt32Vector) VADDV(a, b DenseInt32Vector) Vector {
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
func (r DenseInt32Vector) VaddS(a ConstVector, b ConstScalar) Vector {
  n := r.Dim()
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Add(a.ConstAt(i), b)
  }
  return r
}
func (r DenseInt32Vector) VADDS(a DenseInt32Vector, b Int32) Vector {
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
func (r DenseInt32Vector) VsubV(a, b ConstVector) Vector {
  n := r.Dim()
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Sub(a.ConstAt(i), b.ConstAt(i))
  }
  return r
}
func (r DenseInt32Vector) VSUBV(a, b DenseInt32Vector) Vector {
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
func (r DenseInt32Vector) VsubS(a ConstVector, b ConstScalar) Vector {
  n := r.Dim()
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Sub(a.ConstAt(i), b)
  }
  return r
}
func (r DenseInt32Vector) VSUBS(a DenseInt32Vector, b Int32) Vector {
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
func (r DenseInt32Vector) VmulV(a, b ConstVector) Vector {
  n := r.Dim()
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Mul(a.ConstAt(i), b.ConstAt(i))
  }
  return r
}
func (r DenseInt32Vector) VMULV(a, b DenseInt32Vector) Vector {
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
func (r DenseInt32Vector) VmulS(a ConstVector, s ConstScalar) Vector {
  n := r.Dim()
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Mul(a.ConstAt(i), s)
  }
  return r
}
func (r DenseInt32Vector) VMULS(a DenseInt32Vector, s Int32) Vector {
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
func (r DenseInt32Vector) VdivV(a, b ConstVector) Vector {
  n := r.Dim()
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Div(a.ConstAt(i), b.ConstAt(i))
  }
  return r
}
func (r DenseInt32Vector) VDIVV(a, b DenseInt32Vector) Vector {
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
func (r DenseInt32Vector) VdivS(a ConstVector, s ConstScalar) Vector {
  n := r.Dim()
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r.AT(i).Div(a.ConstAt(i), s)
  }
  return r
}
func (r DenseInt32Vector) VDIVS(a DenseInt32Vector, s Int32) Vector {
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
func (r DenseInt32Vector) MdotV(a ConstMatrix, b ConstVector) Vector {
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
  t := 0.0
  for i := 0; i < n; i++ {
    r.AT(i).Reset()
    for j := 0; j < m; j++ {
      t = a.Float64At(i, j)*b.Float64At(j)
      r.AT(i).Add(r.AT(i), ConstFloat64(t))
    }
  }
  return r
}
func (r DenseInt32Vector) MDOTV(a *DenseInt32Matrix, b DenseInt32Vector) Vector {
  n, m := a.Dims()
  if r.Dim() != n || b.Dim() != m {
    panic("matrix/vector dimensions do not match!")
  }
  if n == 0 || m == 0 {
    return r
  }
  if r.AT(0) == b.AT(0) {
    panic("result and argument must be different vectors")
  }
  t := NullInt32()
  for i := 0; i < n; i++ {
    r.AT(i).Reset()
    for j := 0; j < m; j++ {
      t.MUL(a.AT(i, j), b.AT(j))
      r.AT(i).ADD(r.AT(i), t)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Vector matrix product of a and b. The result is stored in r.
func (r DenseInt32Vector) VdotM(a ConstVector, b ConstMatrix) Vector {
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
  t := 0.0
  for i := 0; i < m; i++ {
    r.AT(i).Reset()
    for j := 0; j < n; j++ {
      t = a.Float64At(j)*b.Float64At(j, i)
      r.AT(i).Add(r.AT(i), ConstFloat64(t))
    }
  }
  return r
}
func (r DenseInt32Vector) VDOTM(a DenseInt32Vector, b *DenseInt32Matrix) Vector {
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
  t := NullInt32()
  for i := 0; i < m; i++ {
    r.AT(i).Reset()
    for j := 0; j < n; j++ {
      t.MUL(a.AT(j), b.AT(j, i))
      r.AT(i).ADD(r.AT(i), t)
    }
  }
  return r
}
