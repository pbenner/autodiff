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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/givensRotation"
import   "github.com/pbenner/autodiff/algorithm/hessenbergReduction"
import   "github.com/pbenner/autodiff/algorithm/householder"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type ComputeU struct {
  Value bool
}

type InSitu struct {
  Hessenberg  hessenbergReduction.InSitu
  InitializeH bool
  InitializeU bool
  H    Matrix
  U    Matrix
  S    Scalar
  Beta Scalar
  Nu   DenseVector
  X    DenseVector
  T    Scalar
  T1   Scalar
  T2   Scalar
  T3   Scalar
  T4   DenseVector
}

/* -------------------------------------------------------------------------- */

func QRstep(h, u Matrix, inSitu *InSitu) {

  n, _ := h.Dims()

  c  := inSitu.S
  s  := inSitu.T
  t1 := inSitu.T1
  t2 := inSitu.T2
  t3 := inSitu.T3

  // shift
  t3.Set(h.At(n-1, n-1))
  for i := 0; i < n; i++ {
    g := h.At(i, i)
    g.Sub(g, t3)
  }
  for i := 0; i < n-1; i++ {
    givensRotation.Run(h.At(i, i), h.At(i+1, i), c, s)
    // multiply with Givens matrix (G H)
    givensRotation.ApplyHessenbergLeft(h, c, s, i, i+1, t1, t2)
    // multiply with Givens matrix (H G)
    givensRotation.ApplyHessenbergRight(h, c, s, i, i+1, t1, t2)
    if u != nil {
      givensRotation.ApplyRight(u, c, s, i, i+1, t1, t2)
    }
  }
  // shift
  for i := 0; i < n; i++ {
    g := h.At(i, i)
    g.Add(g, t3)
  }
}

func francisQRstep(h, u Matrix, inSitu *InSitu) {

  n, _ := h.Dims()

  s  := inSitu.S
  t  := inSitu.T
  x  := inSitu.X
  t1 := inSitu.T1
  t2 := inSitu.T2
  t3 := inSitu.T3
  t4 := inSitu.T4

  beta := inSitu.Beta
  nu   := inSitu.Nu

  h11 := h.At(n-2,n-2)
  h12 := h.At(n-2,n-1)
  h21 := h.At(n-1,n-2)
  h22 := h.At(n-1,n-1)

  s .Add(h11, h22)
  t1.Mul(h11, h22)
  t2.Mul(h12, h21)
  t .Sub(t1 , t2)

  h11 = h.At(0,0)
  h12 = h.At(0,1)
  h21 = h.At(1,0)
  h22 = h.At(1,1)

  t1  .Mul(h11 , h11)
  t2  .Mul(h12 , h21)
  t3  .Mul(s   , h11)
  x[0].Add(t1  , t2)
  x[0].Sub(x[0], t3)
  x[0].Add(x[0], t)

  x[1].Add(h11 , h22)
  x[1].Sub(x[1], s)
  x[1].Mul(x[1], h21)

  x[2].Mul(h21 , h.At(2,1))

  for k := 0; k < n-2; k++ {
    q := 1
    r := n
    if q < k {
      q = k
    }
    if r > k+4 {
      r = k+4
    }
    householder.Run(x, beta, nu, t1, t2, t3)
    {
      h := h.Slice(k, k+3, q-1, n)
      householder.ApplyLeft(h, beta, nu, t4[q-1:n], t1)
    }
    {
      h := h.Slice(0, r, k, k+3)
      householder.ApplyRight(h, beta, nu, t4[0:r], t1)
    }
    if u != nil {
      u := u.Slice(0, r, k, k+3)
      householder.ApplyRight(u, beta, nu, t4, t1)
    }
    x[0].Set(h.At(k+1, k))
    x[1].Set(h.At(k+2, k))
    if k < n-3 {
      x[2].Set(h.At(k+3, k))
    }
  }
  householder.Run(x[0:2], beta, nu[0:2], t1, t2, t3)
  {
    h := h.Slice(n-2, n, n-3, n)
    householder.ApplyLeft(h, beta, nu[0:2], t4[n-3:n], t1)
  }
  {
    h := h.Slice(0, n, n-2, n)
    householder.ApplyRight(h, beta, nu[0:2], t4, t1)
  }
  if u != nil {
    u := u.Slice(0, n, n-2, n)
    householder.ApplyRight(u, beta, nu[0:2], t4, t1)
  }
}

/* -------------------------------------------------------------------------- */

func qrAlgorithm(inSitu *InSitu, epsilon float64) (Matrix, Matrix, error) {

  h    := inSitu.H
  u    := inSitu.U
  n, _ := h.Dims()

  if h_, u_, err := hessenbergReduction.Run(h, &inSitu.Hessenberg, hessenbergReduction.ComputeU{u != nil}); err != nil {
    return nil, nil, err
  } else {
    h = h_
    u = u_
  }

  for n > 2 {
    if v := h.At(n-1, n-2); math.Abs(v.GetValue()) < epsilon {
      n--
    } else {
      h := h.Slice(0,n,0,n)
      francisQRstep(h, u, inSitu)
    }
  }
  return h, u, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  if n != m {
    return nil, nil, fmt.Errorf("`a' must be a square matrix")
  }
  inSitu   := &InSitu{}
  epsilon  := 1e-18
  computeU := false

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case ComputeU:
      computeU = tmp.Value
    case Epsilon:
      epsilon = tmp.Value
    case *InSitu:
      inSitu = tmp
    case InSitu:
      panic("InSitu must be passed by reference")
    }
  }
  if inSitu.H == nil {
    inSitu.H = a.CloneMatrix()
    inSitu.Hessenberg.H = inSitu.H
  } else {
    if n1, m1 := inSitu.H.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
    // initialize h if necessary
    if inSitu.H != a && inSitu.InitializeH {
      inSitu.H.Set(a)
    }
    inSitu.Hessenberg.H = inSitu.H
  }
  if computeU {
    if inSitu.U == nil {
      inSitu.U = NullMatrix(t, n, n)
      inSitu.Hessenberg.U = inSitu.U
    } else {
      if n1, m1 := inSitu.U.Dims(); n1 != n || m1 != m {
        return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
      }
      inSitu.Hessenberg.U = inSitu.U
    }
  } else {
    inSitu.U = nil
  }
  if inSitu.X == nil {
    inSitu.X = NullDenseVector(t, 3)
  }
  if inSitu.Beta == nil {
    inSitu.Beta = NullScalar(t)
  }
  if inSitu.Nu == nil {
    inSitu.Nu = NullDenseVector(t, 3)
  }
  if inSitu.S == nil {
    inSitu.S = NullScalar(t)
  }
  if inSitu.T == nil {
    inSitu.T = NullScalar(t)
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
  if inSitu.Hessenberg.T1 == nil {
    inSitu.Hessenberg.T1 = inSitu.T1
  }
  if inSitu.Hessenberg.T2 == nil {
    inSitu.Hessenberg.T2 = inSitu.T2
  }
  if inSitu.Hessenberg.T3 == nil {
    inSitu.Hessenberg.T3 = inSitu.T3
  }
  return qrAlgorithm(inSitu, epsilon)
}

/* -------------------------------------------------------------------------- */

func Eigenvalues(a Matrix, args... interface{}) (Vector, error) {
  h, _, err := Run(a, args...)
  if err != nil {
    return nil, err
  }
  eigenvalues := h.Diag()
  eigenvalues  = eigenvalues.SortVector(false)

  return eigenvalues, nil
}

func Eigensystem(a Matrix, args... interface{}) (Vector, Matrix, error) {
  // check if a is symmetric
  if !a.IsSymmetric(1e-12) {
    return nil, nil, fmt.Errorf("for computing eigenvectors `a' must be symmetric")
  }
  h, u, err := Run(a, args...)
  if err != nil {
    return nil, nil, err
  }
  return h.Diag(), u, nil
}
