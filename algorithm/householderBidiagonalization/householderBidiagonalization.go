/* Copyright (C) 2017 Philipp Benner
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

package householderBidiagonalization

/* -------------------------------------------------------------------------- */

import   "fmt"
//import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type InSitu struct {
  X     DenseVector
  Beta  Scalar
  Mu    Scalar
  Nu    DenseVector
  Sigma Scalar
  C1    Scalar
  T1    Scalar
  T2   *DenseMatrix
}

/* -------------------------------------------------------------------------- */

func house(x DenseVector, inSitu *InSitu) (Vector, Scalar) {
  n     := x.Dim()
  mu    := inSitu.Mu
  nu    := inSitu.Nu[0:n]
  nu.At(0).SetValue(1.0)
  beta  := inSitu.Beta
  sigma := inSitu.Sigma
  sigma.SetValue(0.0)
  t     := inSitu.T1

  for i := 1; i < n; i++ {
    nu.At(i).Set(x.At(i))
    t.Mul(x.At(i), x.At(i))
    sigma.Add(sigma, t)
  }
  if sigma.GetValue() == 0.0 {
    beta.SetValue(0.0)
  } else {
    nu0 := nu.At(0)
    mu.Mul(x.At(0), x.At(0))
    mu.Add(mu, sigma)
    mu.Sqrt(mu)
    if x.At(0).GetValue() <= 0.0 {
      nu0.Sub(x.At(0), mu)
    } else {
      nu0.Add(x.At(0), mu)
      nu0.Div(sigma, nu0)
      nu0.Neg(nu0)
    }
    beta.Mul(nu0, nu0)
    beta.Add(beta, sigma)
    beta.Div(nu0, beta)
    beta.Mul(beta, nu0)
    beta.Add(beta, beta)
    t.Set(nu0)
    nu.VdivS(nu, t)
  }
  return nu, beta
}

func houseCol(j int, inSitu *InSitu) (Vector, Scalar) {
  A := inSitu.A
  x := inSitu.X

  n, _ := A.Dims()
  for k := j; k < n; k++ {
    x.At(k).Set(A.At(k, j))
  }
  return house(x[j:n], inSitu)
}

func houseRow(j int, inSitu *InSitu) (Vector, Scalar) {
  A := inSitu.A
  x := inSitu.X

  _, n := A.Dims()
  for k := j+1; k < n; k++ {
    x.At(k).Set(A.At(j, k))
  }
  return house(x[j+1:n], inSitu)
}

func householderBidiagonalization(A Matrix, inSitu *InSitu, epsilon float64) (Matrix, error) {

  c1 := inSitu.C1
  T  := inSitu.T2

  m, n := A.Dims()

  for j := 0; j < n; j++ {

    a := A.Slice(j, m  , j, n  )
    t := T.Slice(0, m-j, 0, m-j)

    nu, beta := houseCol(A, j, inSitu)

    // compute (I - beta nu nu^T)
    for j1 := 0; j1 < m-j; j1++ {
      for j2 := 0; j2 < m-j; j2++ {
        s := t.At(j1, j2)
        s.Mul(nu.At(j1), nu.At(j2))
        s.Mul(s, beta)
        if j1 == j2 {
          s.Sub(c1, s)
        } else {
          s.Neg(s)
        }
      }
    }
    // compute (I - beta nu nu^T) A(j:m, j:n)
    a.MdotM(t, a)
    // A(j+1:m, j) = nu(2:m-j+1)
    for j1 := 1; j1 < m-j; j1++ {
      a.At(j1, 0).Set(nu.At(j1))
    }

    if j < n - 2 {

      a := A.Slice(j, m    , j+1, n)
      t := T.Slice(0, n-j-1, 0  , n-j-1)

      nu, beta := houseRow(A, j, inSitu)
      
      // compute (I - beta nu nu^T)
      for j1 := 0; j1 < n-j-1; j1++ {
        for j2 := 0; j2 < n-j-1; j2++ {
          s := t.At(j1, j2)
          s.Mul(nu.At(j1), nu.At(j2))
          s.Mul(s, beta)
          if j1 == j2 {
            s.Sub(c1, s)
          } else {
            s.Neg(s)
          }
        }
      }
      // compute A(j:m, j+1:n) (I - beta nu nu^T)
      a.MdotM(a, t)
      // A(j, j+2:n) = nu(2:n-j)^T
      for j2 := 1; j2 < n-j-1; j2++ {
        a.At(0, j2).Set(nu.At(j2))
      }
    }
  }
  return A, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, error) {

  m, n := a.Dims()
  t    := a.ElementType()

  if m < n {
    return nil, fmt.Errorf("`a' has invalid dimensions")
  }
  inSitu  := &InSitu{}
  epsilon := 1e-18

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case Epsilon:
      epsilon = tmp.Value
    case *InSitu:
      inSitu = tmp
    case InSitu:
      panic("InSitu must be passed by reference")
    }
  }
  if inSitu.X == nil {
    inSitu.X = NullDenseVector(t, m)
  }
  if inSitu.Beta == nil {
    inSitu.Beta = NullScalar(t)
  }
  if inSitu.Mu == nil {
    inSitu.Mu = NullScalar(t)
  }
  if inSitu.Nu == nil {
    inSitu.Nu = NullDenseVector(t, m)
  }
  if inSitu.Sigma == nil {
    inSitu.Sigma = NullScalar(t)
  }
  if inSitu.C1 == nil {
    inSitu.C1 = NewScalar(t, 1.0)
  }
  if inSitu.T1 == nil {
    inSitu.T1 = NullScalar(t)
  }
  if inSitu.T2 == nil {
    inSitu.T2 = NullDenseMatrix(t, m, m)
  }
  return householderBidiagonalization(a, inSitu, epsilon)
}
