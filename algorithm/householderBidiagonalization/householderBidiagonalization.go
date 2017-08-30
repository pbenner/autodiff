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
import   "github.com/pbenner/autodiff/algorithm/householder"

/* -------------------------------------------------------------------------- */

type ComputeU struct {
  Value bool
}

type ComputeV struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type InSitu struct {
  A     Matrix
  U     Matrix
  V     Matrix
  X     DenseVector
  Beta  Scalar
  Nu    DenseVector
  C1    Scalar
  T1    Scalar
  T2    Scalar
  T3    Scalar
  T4    DenseVector
}

/* -------------------------------------------------------------------------- */

func houseCol(j int, inSitu *InSitu) (Vector, Scalar) {
  A    := inSitu.A
  x    := inSitu.X
  beta := inSitu.Beta
  nu   := inSitu.Nu
  t1   := inSitu.T1
  t2   := inSitu.T2
  t3   := inSitu.T3

  n, _ := A.Dims()
  for k := j; k < n; k++ {
    x.At(k).Set(A.At(k, j))
  }
  return householder.Run(x[j:n], beta, nu[j:n], t1, t2, t3)
}

func houseRow(j int, inSitu *InSitu) (Vector, Scalar) {
  A := inSitu.A
  x := inSitu.X
  beta := inSitu.Beta
  nu   := inSitu.Nu
  t1   := inSitu.T1
  t2   := inSitu.T2
  t3   := inSitu.T3

  _, n := A.Dims()
  for k := j+1; k < n; k++ {
    x.At(k).Set(A.At(j, k))
  }
  return householder.Run(x[j+1:n], beta, nu[j+1:n], t1, t2, t3)
}

/* -------------------------------------------------------------------------- */

func householderBidiagonalization(inSitu *InSitu, epsilon float64) (Matrix, Matrix, Matrix, error) {

  A := inSitu.A
  U := inSitu.U
  V := inSitu.V
  t := inSitu.T4

  m, n := A.Dims()

  for j := 0; j < n; j++ {

    a := A.Slice(j, m, j, n)

    nu, beta := houseCol(j, inSitu)

    // compute (I - beta nu nu^T) A(j:m, j:n)
    householder.RunApplyLeft(a, beta, nu, t[j:n], inSitu.T1)
    // accumulate U
    if U != nil {
      nu := inSitu.Nu
      if j > 0 {
        nu.At(j-1).SetValue(0.0)
      }
      householder.RunApplyRight(U, beta, nu, t[0:m], inSitu.T1)
    }

    if j < n - 2 {

      a := A.Slice(j+0, m, j+1, n)

      nu, beta := houseRow(j, inSitu)

      // compute A(j:m, j+1:n) (I - beta nu nu^T)
      householder.RunApplyRight(a, beta, nu, t[j:m], inSitu.T1)
      // accumulate V
      if V != nil {
        nu := inSitu.Nu
        nu.At(j).SetValue(0.0)
        householder.RunApplyLeft(V, beta, nu[0:n], t[0:n], inSitu.T1)
      }
    }
  }
  return A, U, V, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, Matrix, error) {

  m, n := a.Dims()
  t    := a.ElementType()

  if m < n {
    return nil, nil, nil, fmt.Errorf("`a' has invalid dimensions")
  }
  inSitu   := &InSitu{}
  computeU := false
  computeV := false
  epsilon  := 1.11e-16

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case ComputeU:
      computeU = tmp.Value
    case ComputeV:
      computeV = tmp.Value
    case Epsilon:
      epsilon = tmp.Value
    case *InSitu:
      inSitu = tmp
    case InSitu:
      panic("InSitu must be passed by reference")
    }
  }
  if inSitu.A == nil {
    inSitu.A = a.CloneMatrix()
  } else {
    if inSitu.A != a {
      inSitu.A.Set(a)
    }
  }
  if computeU {
    if inSitu.U == nil {
      inSitu.U = NullDenseMatrix(t, m, m)
    }
    // initialized by householderBidiagonalization
  } else {
    inSitu.U = nil
  }
  if computeV {
    if inSitu.V == nil {
      inSitu.V = NullDenseMatrix(t, n, n)
    }
    // initialized by householderBidiagonalization
  } else {
    inSitu.V = nil
  }
  if inSitu.X == nil {
    inSitu.X = NullDenseVector(t, m)
  }
  if inSitu.Beta == nil {
    inSitu.Beta = NullScalar(t)
  }
  if inSitu.Nu == nil {
    inSitu.Nu = NullDenseVector(t, m)
  }
  if inSitu.C1 == nil {
    inSitu.C1 = NewScalar(t, 1.0)
  }
  if inSitu.T1 == nil {
    inSitu.T1 = NullScalar(t)
  }
  if inSitu.T2 == nil {
    inSitu.T2 = NullScalar(t)
  }
  if inSitu.T3 == nil {
    inSitu.T3 = NullScalar(t)
  }
  if inSitu.T4 == nil {
    inSitu.T4 = NullDenseVector(t, m)
  }
  return householderBidiagonalization(inSitu, epsilon)
}
