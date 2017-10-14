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

func cholesky_DenseBareRealMatrix(A, L *DenseBareRealMatrix, t, s *BareReal) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
  n, _  := A.Dims()

  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.MUL(L.AT(i,k), L.AT(j,k))
        s.ADD(s, t)
      }
      t.SUB(A.AT(i, j), s)
      if i == j {
        if t.GetValue() < 0.0 {
          return nil, nil, errors.New("matrix is not positive definite")
        }
        L.AT(i, j).SQRT(t)
      } else {
        L.AT(i, j).DIV(t, L.AT(j, j))
      }
    }
  }
  return L, nil, nil
}

func choleskyInSitu_DenseBareRealMatrix(A *DenseBareRealMatrix) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
  n, _  := A.Dims()
  t     := NewBareReal(0.0)
  s     := NewBareReal(0.0)
  Aii   := NewBareReal(0.0)

  for i := 0; i < n; i++ {
    Aii.Set(A.AT(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.MUL(A.AT(i,k), A.AT(j,k))
        s.ADD(s, t)
      }
      if i == j {
        t.SUB(Aii, s)
        if t.GetValue() < 0.0 {
          return nil, nil, errors.New("matrix is not positive definite")
        }
        A.AT(j, i).SQRT(t)
      } else {
        t.SUB(A.AT(i, j), s)
        A.AT(i, j).DIV(t, A.AT(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.AT(j, i)
      A.AT(j, i).Set(r)
      r.Reset()
    }
  }
  return A, nil, nil
}
