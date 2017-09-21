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

package cholesky

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "errors"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func cholesky_RealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewReal(0.0)
  s     := NewReal(0.0)
  L     := NullDenseMatrix(RealType, n, n)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.RealMul(L.RealAt(i,k), L.RealAt(j,k))
        s.RealAdd(s, t)
      }
      t.RealSub(A.RealAt(i, j), s)
      if i == j {
        if t.GetValue() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        L.RealAt(i, j).RealSqrt(t)
      } else {
        L.RealAt(i, j).RealDiv(t, L.RealAt(j, j))
      }
    }
  }
  return L, nil
}

func choleskyInSitu_RealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewReal(0.0)
  s     := NewReal(0.0)
  Aii   := NewReal(0.0)

  for i := 0; i < n; i++ {
    Aii.Set(A.RealAt(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.RealMul(A.RealAt(i,k), A.RealAt(j,k))
        s.RealAdd(s, t)
      }
      if i == j {
        t.RealSub(Aii, s)
        if t.GetValue() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        A.RealAt(j, i).RealSqrt(t)
      } else {
        t.RealSub(A.RealAt(i, j), s)
        A.RealAt(i, j).RealDiv(t, A.RealAt(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.RealAt(j, i)
      A.RealAt(j, i).Set(r)
      r.Reset()
    }
  }
  return A, nil
}

/* -------------------------------------------------------------------------- */

func cholesky_BareRealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewBareReal(0.0)
  s     := NewBareReal(0.0)
  L     := NullDenseMatrix(BareRealType, n, n)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.BareRealMul(L.BareRealAt(i,k), L.BareRealAt(j,k))
        s.BareRealAdd(s, t)
      }
      t.BareRealSub(A.BareRealAt(i, j), s)
      if i == j {
        if t.GetValue() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        L.BareRealAt(i, j).BareRealSqrt(t)
      } else {
        L.BareRealAt(i, j).BareRealDiv(t, L.BareRealAt(j, j))
      }
    }
  }
  return L, nil
}

func choleskyInSitu_BareRealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewBareReal(0.0)
  s     := NewBareReal(0.0)
  Aii   := NewBareReal(0.0)

  for i := 0; i < n; i++ {
    Aii.Set(A.BareRealAt(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.BareRealMul(A.BareRealAt(i,k), A.BareRealAt(j,k))
        s.BareRealAdd(s, t)
      }
      if i == j {
        t.BareRealSub(Aii, s)
        if t.GetValue() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        A.BareRealAt(j, i).BareRealSqrt(t)
      } else {
        t.BareRealSub(A.BareRealAt(i, j), s)
        A.BareRealAt(i, j).BareRealDiv(t, A.BareRealAt(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.BareRealAt(j, i)
      A.BareRealAt(j, i).Set(r)
      r.Reset()
    }
  }
  return A, nil
}
