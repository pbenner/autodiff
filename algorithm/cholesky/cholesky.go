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
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  L Matrix
  D Matrix
  S Scalar
  T Scalar
}

type LDL struct {
  Value bool
}

type ForcePD struct {
  Value bool
}

func NewInSitu(t ScalarType, n int, ldl bool) InSitu {
  s := InSitu{}
  s.L = NullMatrix(t, n, n)
  s.S = NewScalar(t, 0.0)
  s.T = NewScalar(t, 0.0)
  if ldl {
    s.D = NullMatrix(t, n, n)
  }
  return s
}

/* -------------------------------------------------------------------------- */

func cholesky(A, L Matrix, s, t Scalar) (Matrix, Matrix, error) {
  n, _  := A.Dims()

  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(L.At(i,k), L.At(j,k))
        s.Add(s, t)
      }
      t.Sub(A.At(i, j), s)
      if i == j {
        if t.GetValue() < 0.0 {
          return nil, nil, errors.New("matrix is not positive definite")
        }
        L.At(i, j).Sqrt(t)
      } else {
        L.At(i, j).Div(t, L.At(j, j))
      }
    }
  }
  return L, nil, nil
}

func choleskyLDL(A, L, D Matrix, s, t Scalar) (Matrix, Matrix, error) {
  n, _  := A.Dims()

  c := t

  for j := 0; j < n; j++ {
    // compute diagonal entries
    s.Reset()
    for k := 0; k < j; k++ {
      t.Mul(L.At(j,k), L.At(j,k))
      t.Mul(D.At(k,k), t)
      s.Add(s, t)
    }
    c.Sub(A.At(j, j), s)
    D.At(j,j).Set(c)
    if D.At(j,j).GetValue() <= 0.0 {
      return nil, nil, errors.New("matrix is not positive definite")
    }
    L.At(j,j).SetValue(1.0)
    // compute remaining entries
    for i := j+1; i < n; i++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(L.At(i,k), L.At(j,k))
        t.Mul(D.At(k,k), t)
        s.Add(s, t)
      }
      c.Sub(A.At(i, j), s)
      L.At(i,j).Div(c, D.At(j,j))
    }
  }
  return L, D, nil
}

func choleskyLDLforcePD(A, L, D Matrix, s, t Scalar) (Matrix, Matrix, error) {
  n, _  := A.Dims()

  // compute beta and gamma
  beta  := 0.0
  gamma := math.Inf(-1)
  nu    := math.Max(1, math.Sqrt(float64(n*n - 1)))
  theta := math.Inf(-1)
  xi    := math.Inf(-1)
  delta := 1e-20

  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        if r := math.Abs(A.At(i, i).GetValue()); r > gamma {
          gamma = r
        }
      } else {
        if r := math.Abs(A.At(i, j).GetValue()); r > xi {
          xi = r
        }
      }
    }
  }
  beta = math.Max(gamma, xi/nu)
  beta = math.Max(beta,  1e-20)
  beta = math.Sqrt(beta)

  // loop over columns
  for j := 0; j < n; j++ {
    L.At(j,j).SetValue(1.0)
    // compute c_jj (stored temporarily in d_j)
    s.Reset()
    for k := 0; k < j; k++ {
      t.Mul(L.At(j,k), L.At(j,k))
      t.Mul(D.At(k,k), t)
      s.Add(s, t)
    }
    c_jj := D.At(j,j)
    c_jj.Sub(A.At(j,j), s)
    // reset theta_j
    theta = math.Inf(-1)
    // compute c_ij and theta_j
    for i := j+1; i < n; i++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(L.At(i,k), L.At(j,k))
        t.Mul(D.At(k,k), t)
        s.Add(s, t)
      }
      // result: L(i,j) <- c_ij
      L.At(i,j).Sub(A.At(i,j), s)
      // update theta_j
      if r := math.Abs(L.At(i,j).GetValue()); r > theta {
        theta = r
      }
    }
    // compute d_j = max(|c_jj|, (theta_j/beta)^2, delta)
    if j != n-1 {
      D.At(j,j).SetValue(
        math.Max(math.Max(math.Abs(c_jj.GetValue()), math.Pow((theta/beta), 2.0)), delta))
    } else {
      D.At(j,j).SetValue(
        math.Max(math.Abs(c_jj.GetValue()), delta))
    }
    // compute l_ij = c_ij/d_j
    for i := j+1; i < n; i++ {
      L.At(i,j).Div(L.At(i,j), D.At(j,j))
    }
  }
  return L, D, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {
  n, m := a.Dims()
  if n != m {
    panic("Cholesky(): Not a square matrix!")
  }
  if n == 0 {
    panic("Cholesky(): Empty matrix!")
  }
  t       := a.ElementType()
  inSitu  := &InSitu{}
  ldl     := false
  forcePD := false

  for _, arg := range args {
    switch a := arg.(type) {
    case LDL:
      ldl = a.Value
    case ForcePD:
      forcePD = a.Value
    case *InSitu:
      inSitu = a
    case InSitu:
      panic("InSitu must be passed by reference")
    default:
      panic("Cholesky(): Invalid optional argument!")
    }
  }
  // allocate memory
  if inSitu.L == nil {
    inSitu.L = NullMatrix(t, n, n)
  }
  if ldl {
    if inSitu.D == nil {
      inSitu.D = NullMatrix(t, n, n)
    } else {
      for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
          inSitu.D.At(i,j).SetValue(0.0)
        }
      }
    }
  }
  if inSitu.S == nil {
    inSitu.S = NewScalar(t, 0.0)
  }
  if inSitu.T == nil {
    inSitu.T = NewScalar(t, 0.0)
  }
  // if ad, ok := a.(*DenseMatrix); ok {
  //   t := a.ElementType()
  //   if t == RealType && inSitu == true {
  //     return choleskyInSitu_RealDense(ad)
  //   } else if t == RealType && inSitu == false {
  //     return cholesky_RealDense(ad)
  //   } else if t == BareRealType && inSitu == true {
  //     return choleskyInSitu_BareRealDense(ad)
  //   } else if t == BareRealType && inSitu == false {
  //     return cholesky_BareRealDense(ad)
  //   }
  // }
  if ldl {
    if forcePD {
      return choleskyLDLforcePD(a, inSitu.L, inSitu.D, inSitu.S, inSitu.T)
    } else {
      return choleskyLDL(a, inSitu.L, inSitu.D, inSitu.S, inSitu.T)
    }
  } else {
    return cholesky(a, inSitu.L, inSitu.S, inSitu.T)
  }
}
