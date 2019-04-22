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
// Test if elements in a equal elements in b.
func (a *SparseRealVector) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for entry := range a.JOINT_RANGE(b) {
    if entry.Value1 != nil && entry.Value2 != nil {
      if !entry.Value1.Equals(entry.Value2, epsilon) {
        return false
      }
    } else {
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
  for entry := range r.JOINT_RANGE3(a, b) {
    s_r := entry.Value1
    s_a := entry.Value2
    s_b := entry.Value3
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Add(s_a, s_b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VaddS(a ConstVector, b ConstScalar) Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE(a) {
    s_r := entry.Value1
    s_a := entry.Value2
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Add(s_a, b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of two vectors. The result is stored in r.
func (r *SparseRealVector) VsubV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE3(a, b) {
    s_r := entry.Value1
    s_a := entry.Value2
    s_b := entry.Value3
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Sub(s_a, s_b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substractor of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VsubS(a ConstVector, b ConstScalar) Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE(a) {
    s_r := entry.Value1
    s_a := entry.Value2
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Sub(s_a, b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two vectors. The result is stored in r.
func (r *SparseRealVector) VmulV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE3(a, b) {
    s_r := entry.Value1
    s_a := entry.Value2
    s_b := entry.Value3
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Mul(s_a, s_b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VmulS(a ConstVector, s ConstScalar) Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE(a) {
    s_r := entry.Value1
    s_a := entry.Value2
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Mul(s_a, s)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of two vectors. The result is stored in r.
func (r *SparseRealVector) VdivV(a, b ConstVector) Vector {
  if n := r.Dim(); a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE3(a, b) {
    s_r := entry.Value1
    s_a := entry.Value2
    s_b := entry.Value3
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Div(s_a, s_b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of a vector and a scalar. The result is stored in r.
func (r *SparseRealVector) VdivS(a ConstVector, s ConstScalar) Vector {
  if r.Dim() != a.Dim() {
    panic("vector dimensions do not match")
  }
  for entry := range r.JOINT_RANGE(a) {
    s_r := entry.Value1
    s_a := entry.Value2
    if s_r == nil {
      s_r = r.AT(entry.Index)
    }
    s_r.Div(s_a, s)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Matrix vector product of a and b. The result is stored in r.
func (r *SparseRealVector) MdotV(a Matrix, b ConstVector) Vector {
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
      t.Mul(a.At(i, j), b.ConstAt(j))
      r.AT(i).ADD(r.AT(i), t)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Vector matrix product of a and b. The result is stored in r.
func (r *SparseRealVector) VdotM(a ConstVector, b Matrix) Vector {
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
      t.Mul(a.ConstAt(j), b.At(j, i))
      r.AT(i).ADD(r.AT(i), t)
    }
  }
  return r
}
