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
//import "math"
/* -------------------------------------------------------------------------- */
// Test if elements in a equal elements in b.
func (a *SparseFloat64Vector) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for it := a.JOINT_ITERATOR(b); it.Ok(); it.Next() {
    s1, s2 := it.GET()
    if s1.ptr == nil {
      return false
    }
    if !s1.Equals(s2, epsilon) {
      return false
    }
  }
  return true
}
func (a *SparseFloat64Vector) EQUALS(b *SparseFloat64Vector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for it := a.JOINT_ITERATOR_(b); it.Ok(); it.Next() {
    s1, s2 := it.GET()
    if s1.ptr == nil {
      return false
    }
    if s2.ptr == nil {
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
func (r *SparseFloat64Vector) VaddV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    s_r.Add(s_a, s_b)
  }
  return r
}
func (r *SparseFloat64Vector) VADDV(a, b *SparseFloat64Vector) *SparseFloat64Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    switch {
    case s_a.ptr == nil && s_b.ptr == nil:
      s_r.SetFloat64(0.0)
    case s_b.ptr == nil:
      s_r.SET(s_a)
    case s_a.ptr == nil:
      s_r.SET(s_b)
    default:
      s_r.ADD(s_a, s_b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r *SparseFloat64Vector) VaddS(a ConstVector, b ConstScalar) Vector {
  n := r.Dim()
  if n != a.Dim() {
    panic("vector dimensions do not match")
  }
  for i := 0; i < n; i++ {
    r.AT(i).Add(a.ConstAt(i), b)
  }
  return r
}
func (r *SparseFloat64Vector) VADDS(a *SparseFloat64Vector, b Float64) *SparseFloat64Vector {
  r.VaddS(a, b)
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of two vectors. The result is stored in r.
func (r *SparseFloat64Vector) VsubV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    s_r.Sub(s_a, s_b)
  }
  return r
}
func (r *SparseFloat64Vector) VSUBV(a, b *SparseFloat64Vector) *SparseFloat64Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    switch {
    case s_a.ptr == nil && s_b.ptr == nil:
      s_r.SetFloat64(0.0)
    case s_b.ptr == nil:
      s_r.SET(s_a)
    case s_a.ptr == nil:
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
func (r *SparseFloat64Vector) VsubS(a ConstVector, b ConstScalar) Vector {
  n := r.Dim()
  if n != a.Dim() {
    panic("vector dimensions do not match")
  }
  for i := 0; i < n; i++ {
    r.AT(i).Sub(a.ConstAt(i), b)
  }
  return r
}
func (r *SparseFloat64Vector) VSUBS(a *SparseFloat64Vector, b Float64) *SparseFloat64Vector {
  r.VsubS(a, b)
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two vectors. The result is stored in r.
func (r *SparseFloat64Vector) VmulV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      continue
    }
    s_r.Mul(s_a, s_b)
  }
  return r
}
func (r *SparseFloat64Vector) VMULV(a, b *SparseFloat64Vector) *SparseFloat64Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT3_ITERATOR_(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    switch {
    case s_a.ptr == nil || s_b.ptr == nil:
      s_r.SetFloat64(0.0)
    default:
      s_r.MUL(s_a, s_b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r *SparseFloat64Vector) VmulS(a ConstVector, b ConstScalar) Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    if s_r.ptr == nil {
      continue
    }
    s_r.Mul(s_a, b)
  }
  return r
}
func (r *SparseFloat64Vector) VMULS(a *SparseFloat64Vector, b Float64) *SparseFloat64Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    if s_a.ptr == nil {
      s_r.SetFloat64(0.0)
    } else {
      s_r.MUL(s_a, b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of two vectors. The result is stored in r.
func (r *SparseFloat64Vector) VdivV(a, b ConstVector) Vector {
  n := r.Dim()
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < n; i++ {
    c1 := a.ConstAt(i)
    c2 := b.ConstAt(i)
    if c1.GetFloat64() != 0.0 || c2.GetFloat64() == 0.0 {
      r.At(i).Div(c1, c2)
    } else {
      if r.ConstAt(i).GetFloat64() != 0.0 {
        r.At(i).Reset()
      }
    }
  }
  return r
}
func (r *SparseFloat64Vector) VDIVV(a, b *SparseFloat64Vector) *SparseFloat64Vector {
  r.VdivV(a, b)
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of a vector and a scalar. The result is stored in r.
func (r *SparseFloat64Vector) VdivS(a ConstVector, b ConstScalar) Vector {
  n := r.Dim()
  if n != a.Dim() {
    panic("vector dimensions do not match")
  }
  if b.GetFloat64() == 0.0 {
    for i := 0; i < n; i++ {
      r.At(i).Div(a.ConstAt(i), b)
    }
  } else {
    for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
      s_r := it.s1
      s_a := it.s2
      if s_r.ptr == nil {
        s_r = r.AT(it.Index())
      }
      s_r.Div(s_a, b)
    }
  }
  return r
}
func (r *SparseFloat64Vector) VDIVS(a *SparseFloat64Vector, b Float64) *SparseFloat64Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for it := r.JOINT_ITERATOR_(a); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    if s_a.ptr == nil {
      s_r.SetFloat64(0.0)
    } else {
      s_r.DIV(s_a, b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Matrix vector product of a and b. The result is stored in r.
func (r *SparseFloat64Vector) MdotV(a ConstMatrix, b ConstVector) Vector {
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
  t := NullFloat64()
  for i := 0; i < n; i++ {
    r.AT(i).Reset()
  }
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    i, j := it.Index()
    s := r.AT(i)
    t.Mul(it.GetConst(), b.ConstAt(j))
    s.ADD(s, t)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Vector matrix product of a and b. The result is stored in r.
func (r *SparseFloat64Vector) VdotM(a ConstVector, b ConstMatrix) Vector {
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
  t := NullFloat64()
  for i := 0; i < n; i++ {
    r.AT(i).Reset()
  }
  for it := b.ConstIterator(); it.Ok(); it.Next() {
    i, j := it.Index()
    s := r.AT(j)
    t.Mul(a.ConstAt(i), it.GetConst())
    s.ADD(s, t)
  }
  return r
}
