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
//import "fmt"
/* -------------------------------------------------------------------------- */
// True if matrix a equals b.
func (a *SparseIntMatrix) Equals(b ConstMatrix, epsilon float64) bool {
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("MEqual(): matrix dimensions do not match!")
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
/* -------------------------------------------------------------------------- */
// Element-wise addition of two matrices. The result is stored in r.
func (r *SparseIntMatrix) MaddM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
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
/* -------------------------------------------------------------------------- */
// Add scalar b to all elements of a. The result is stored in r.
func (r *SparseIntMatrix) MaddS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Add(a.ConstAt(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of two matrices. The result is stored in r.
func (r *SparseIntMatrix) MsubM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
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
/* -------------------------------------------------------------------------- */
// Substract b from all elements of a. The result is stored in r.
func (r *SparseIntMatrix) MsubS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Sub(a.ConstAt(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two matrices. The result is stored in r.
func (r *SparseIntMatrix) MmulM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for it := r.JOINT3_ITERATOR(a, b); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    s_b := it.s3
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    switch {
    case s_a == nil || s_b == nil:
      s_r.SetInt(0)
    default:
      s_r.Mul(s_a, s_b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Multiply all elements of a with b. The result is stored in r.
func (r *SparseIntMatrix) MmulS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for it := r.JOINT_ITERATOR(a); it.Ok(); it.Next() {
    s_r := it.s1
    s_a := it.s2
    if s_r.ptr == nil {
      s_r = r.AT(it.Index())
    }
    s_r.Mul(s_a, b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of two matrices. The result is stored in r.
func (r *SparseIntMatrix) MdivM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      c1 := a.ConstAt(i, j)
      c2 := b.ConstAt(i, j)
      if c1.GetInt() != int(0) || c2.GetInt() == int(0) {
        r.At(i, j).Div(c1, c2)
      } else {
        if r.ConstAt(i, j).GetInt() != 0.0 {
          r.At(i, j).Reset()
        }
      }
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Divide all elements of a by b. The result is stored in r.
func (r *SparseIntMatrix) MdivS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  if b.GetInt() == int(0) {
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        r.At(i, j).Div(a.ConstAt(i, j), b)
      }
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
/* -------------------------------------------------------------------------- */
// Matrix product of a and b. The result is stored in r.
func (r *SparseIntMatrix) MdotM(a, b ConstMatrix) Matrix {
  n , m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m2 != m || m1 != n2 {
    panic("matrix dimensions do not match!")
  }
  if r.storageLocation() == a.storageLocation() ||
     r.storageLocation() == b.storageLocation() {
    panic("result and argument must be different matrices")
  }
  t1 := NullScalar(r.ElementType())
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    i, j := it.Index()
    for is := b.ConstIteratorFrom(j, 0); is.Ok(); is.Next() {
      p, q := is.Index()
      if p != j {
        break
      }
      t1.Mul(it.GetConst(), is.GetConst())
      r_ := r.At(i, q)
      r_.Add(r_, t1)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Outer product of two vectors. The result is stored in r.
func (r *SparseIntMatrix) Outer(a, b ConstVector) Matrix {
  n, m := r.Dims()
  if a.Dim() != n || b.Dim() != m {
    panic("matrix/vector dimensions do not match!")
  }
  for it := r.Iterator(); it.Ok(); it.Next() {
    it.Get().Reset()
  }
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    i := it.Index()
    p := it.GetConst()
    for is := b.ConstIterator(); is.Ok(); is.Next() {
      j := is.Index()
      q := is.GetConst()
      r.At(i, j).Mul(p, q)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Compute the Jacobian of f at x_. The result is stored in r.
func (r *SparseIntMatrix) Jacobian(f func(ConstVector) ConstVector, x_ MagicVector) Matrix {
  n, m := r.Dims()
  x := x_.CloneMagicVector()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  // reallocate matrix if dimensions do not match
  if r == nil || x.Dim() != m || y.Dim() != n {
     n = y.Dim()
     m = x.Dim()
    *r = *NullSparseIntMatrix(n, m)
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      if s := y.ConstAt(i).GetDerivative(j); s != 0.0 {
        r.At(i, j).SetFloat64(s)
      }
    }
  }
  return r
}
// Compute the Hessian of f at x_. The result is stored in r.
func (r *SparseIntMatrix) Hessian(f func(ConstVector) ConstScalar, x_ MagicVector) Matrix {
  n, m := r.Dims()
  // reallocate matrix if dimensions do not match
  if r == nil || x_.Dim() != n || n != m {
     n = x_.Dim()
     m = x_.Dim()
    *r = *NullSparseIntMatrix(n, m)
  }
  x := x_.CloneMagicVector()
  x.Variables(2)
  // evaluate function
  y := f(x)
  // copy second derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      if s := y.GetHessian(i, j); s != 0.0 {
        r.At(i, j).SetFloat64(s)
      }
    }
  }
  return r
}
