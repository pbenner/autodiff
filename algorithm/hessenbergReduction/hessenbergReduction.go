/* Copyright (C) 2015 Philipp Benner
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

package hessenbergReduction

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  InitializeH bool
  InitializeV bool
  H Matrix
  V Matrix
  X Vector
  U Vector
  S Scalar
}

func NewInSitu(t ScalarType, n int) InSitu {
  s := InSitu{}
  s.InitializeH = true
  s.InitializeV = true
  s.H = NullMatrix(t, n, n)
  s.V = NullMatrix(t, n, n)
  s.X = NullVector(t, n)
  s.U = NullVector(t, n)
  s.S = NullScalar(t)
  return s
}

/* -------------------------------------------------------------------------- */

func fu(x, u Vector, s Scalar) (Vector, bool) {
  // s = ||x||
  s.Vnorm(x)
  if s.GetValue() == 0.0 {
    // diagonal is already zero
    return u, false
  }
  // s = -sign(x.At(0)) ||x||
  if x.At(0).GetValue() > 0.0 {
    s.Neg(s)
  }
  // u = x - s e_1
  u.At(0).Sub(x.At(0), s)
  for i := 1; i < x.Dim(); i++ {
    u.At(i).Set(x.At(i))
  }
  // s = ||u||
  s.Vnorm(u)
  // u = u/s
  u.VdivS(u, s)
  return u, true
}

func hessenbergReduction(a, v Matrix, x, u Vector, s Scalar) (Matrix, Matrix, error) {
  n, _ := a.Dims()

  for k := 0; k < n-2; k++ {
    // copy column below main diagonal from A to x,
    // x = (A[k+1,k], A[k+2,k], ..., A[n-1,k])
    for i := k+1; i < n; i++ {
      x.At(i).Set(a.At(i, k))
    }
    if _, ok := fu(x.Slice(k+1,n), u.Slice(k+1,n), s); !ok {
      continue
    }
    // A <- P_k A = A - 2 u (u^t A)
    // i) compute u^t A and store it in x
    for j := k; j < n; j++ {
      x.At(j).Reset()
      for i := k+1; i < n; i++ {
        s.Mul(u.At(i), a.At(i, j))
        x.At(j).Add(x.At(j), s)
      }
    }
    // ii) compute A - 2 u (u^t A) = A - 2 u x^t
    for i := k+1; i < n; i++ {
      for j := k; j < n; j++ {
        s.Mul(u.At(i), x.At(j))
        s.Add(s, s)
        a.At(i, j).Sub(a.At(i, j), s)
      }
    }
    // A <- A P_k = A - 2 (A u) u^t
    // i) compute A u and store it in x
    for i := 0; i < n; i++ {
      x.At(i).Reset()
      for j := k+1; j < n; j++ {
        s.Mul(a.At(i, j), u.At(j))
        x.At(i).Add(x.At(i), s)
      }
    }
    // ii) compute A - 2 (A u) u^t = A - 2 x u^t
    for i := 0; i < n; i++ {
      for j := k+1; j < n; j++ {
        s.Mul(x.At(i), u.At(j))
        s.Add(s, s)
        a.At(i, j).Sub(a.At(i, j), s)
      }
    }
    if v != nil {
      // A <- A P_k = A - 2 (A u) u^t
      // i) compute A u and store it in x
      for i := 0; i < n; i++ {
        x.At(i).Reset()
        for j := k+1; j < n; j++ {
          s.Mul(v.At(i, j), u.At(j))
          x.At(i).Add(x.At(i), s)
        }
      }
      // ii) compute A - 2 (A u) u^t = A - 2 x u^t
      for i := 0; i < n; i++ {
        for j := k+1; j < n; j++ {
          s.Mul(x.At(i), u.At(j))
          s.Add(s, s)
          v.At(i, j).Sub(v.At(i, j), s)
        }
      }
    }
  }
  return a, v, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  if n != m {
    return nil, nil, fmt.Errorf("`a' must be a square matrix")
  }

  initializeH := true
  initializeV := true

  var h Matrix
  var v Matrix
  var x Vector
  var u Vector
  var s Scalar

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case InSitu:
      initializeH = tmp.InitializeH
      initializeV = tmp.InitializeV
      h = tmp.H
      v = tmp.V
      x = tmp.X
      u = tmp.U
      s = tmp.S
    }
  }
  if h == nil {
    h = a.CloneMatrix()
  } else {
    if n1, m1 := h.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
    if a != h && initializeH {
      h.Set(a)
    }
  }
  if v == nil {
    v = IdentityMatrix(t, n)
  } else {
    if n1, m1 := v.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
    if initializeV {
      v.SetIdentity()
    }
  }
  if x == nil {
    x = NullVector(t, n)
  } else {
    if n1 := x.Dim(); n1 != n {
      return nil, nil, fmt.Errorf("x has invalid dimension (%d instead of %d)", n1, n)
    }
  }
  if u == nil {
    u = NullVector(t, n)
  } else {
    if n1 := u.Dim(); n1 != n {
      return nil, nil, fmt.Errorf("u has invalid dimension (%d instead of %d)", n1, n)
    }
  }
  if s == nil {
    s = NullScalar(t)
  }
  return hessenbergReduction(h, v, x, u, s)
}
