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
import   "github.com/pbenner/autodiff/algorithm/gramSchmidt"
import   "github.com/pbenner/autodiff/algorithm/hessenbergReduction"

/* -------------------------------------------------------------------------- */

type Shift struct {
  Value bool
}

type Epsilon struct {
  Value float64
}

type InSitu struct {
  Hessenberg hessenbergReduction.InSitu
  InitializeH bool
  InitializeU bool
  H Matrix
  U Matrix
  C Scalar
  S Scalar
  T1, T2, T3 Scalar
}

func NewInSitu(t ScalarType, n int) InSitu {
  s := InSitu{}
  s.Hessenberg  = hessenbergReduction.NewInSitu(t, n)
  s.Hessenberg.InitializeH = false
  s.Hessenberg.InitializeV = false
  s.InitializeH = true
  s.InitializeU = true
  s.H  = s.Hessenberg.H
  s.U  = s.Hessenberg.V
  s.C  = NullScalar(t)
  s.S  = NullScalar(t)
  s.T1 = NullScalar(t)
  s.T2 = NullScalar(t)
  s.T3 = NullScalar(t)
  return s
}

/* -------------------------------------------------------------------------- */

func givens(a, b, c, s Scalar) {
  // fake temporary variables
  t1 := s
  t2 := c

  t1.Reset()
  // t1 = a^2 + b^2
  t2.Mul(a, a)
  t1.Add(t1, t2)

  t2.Mul(b, b)
  t1.Add(t1, t2)
  // t1 = sqrt(a^2 + b^2)
  t1.Sqrt(t1)

  c.Div(a, t1)
  s.Div(b, t1)
}

func hessenbergQrAlgorithmStep(h, u Matrix, c, s Scalar, t1, t2, t3 Scalar, n int, shift bool) {

  N, _ := u.Dims()

  if shift {
    t3.Set(h.At(n-1, n-1))
    for i := 0; i < n; i++ {
      g := h.At(i, i)
      g.Sub(g, t3)
    }
  }
  for i := 0; i < n-1; i++ {
    givens(h.At(i, i), h.At(i+1, i), c, s)

    // multiply with Givens matrix (G H)
    for j := 0; j < N; j++ {
      h1 := h.At(i+0, j)
      h2 := h.At(i+1, j)
      // backup h1
      t1.Set(h1)       // t1 = h1
      // update h1
      h1.Mul(c, h1)    // h1 = c h1
      t2.Mul(s, h2)    // t2 = s h2
      h1.Add(h1, t2)   // h1 = c h1 + s h2
      // update h2
      t1.Mul(s, t1)    // t1 =  s h1
      t1.Neg(t1)       // t1 = -s h1
      t2.Mul(c, h2)    // t2 =  c h2
      h2.Add(t1, t2)   // h2 = -s h1 + c h2
    }
    // multiply with Givens matrix (H G)
    for j := 0; j < N; j++ {
      h1 := h.At(j, i+0)
      h2 := h.At(j, i+1)
      // backup h1
      t1.Set(h1)       // t1 = h1
      // update h1
      h1.Mul(c, h1)    // h1 = c h1
      t2.Mul(s, h2)    // t2 = s h2
      h1.Add(h1, t2)   // h1 = c h1 + s h2
      // update h2
      t1.Mul(s, t1)    // t1 =  s h1
      t1.Neg(t1)       // t1 = -s h1
      t2.Mul(c, h2)    // t2 =  c h2
      h2.Add(t1, t2)   // h2 = -s h1 + c h2
    }
    if u != nil {
      for j := 0; j < N; j++ {
        u1 := u.At(j, i+0)
        u2 := u.At(j, i+1)
        // backup u1
        t1.Set(u1)       // t1 = u1
        // update u1
        u1.Mul(c, u1)    // u1 = c u1
        t2.Mul(s, u2)    // t2 = s u2
        u1.Add(u1, t2)   // u1 = c u1 + s u2
        // update u2
        t1.Mul(s, t1)    // t1 =  s u1
        t1.Neg(t1)       // t1 = -s u1
        t2.Mul(c, u2)    // t2 =  c u2
        u2.Add(t1, t2)   // u2 = -s u1 + c u2
      }
    }
  }
  if shift {
    for i := 0; i < n; i++ {
      g := h.At(i, i)
      g.Add(g, t3)
    }
  }
}

func hessenbergQrAlgorithm(inSitu *InSitu, epsilon float64, shift bool) (Matrix, Matrix, error) {

  h  := inSitu.H
  u  := inSitu.U
  c  := inSitu.C
  s  := inSitu.S
  t1 := inSitu.T1
  t2 := inSitu.T2
  t3 := inSitu.T3

  n, _ := h.Dims()

  _, _, err := hessenbergReduction.Run(h, inSitu.Hessenberg)
  if err != nil {
    return nil, nil, err
  }

  for n > 1 {
    if v := h.At(n-1, n-2); math.Abs(v.GetValue()) < epsilon {
      n--
    } else {
      hessenbergQrAlgorithmStep(h, u, c, s, t1, t2, t3, n, shift)
    }
  }

  return h, u, nil
}

/* naive implementation
 * -------------------------------------------------------------------------- */

func qrAlgorithm(a Matrix) (Matrix, Matrix, error) {

  t := a.ElementType()

  n, m := a.Dims()

  a = a.CloneMatrix()
  var b Matrix = NullMatrix(t, n, m)
  var q Matrix = NullMatrix(t, n, m)
  var r Matrix = NullMatrix(t, n, m)

  for {
    // reset values of q and r
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        q.At(i, j).Reset()
        r.At(i, j).Reset()
      }
    }
    q, r, _ = gramSchmidt.Run(a, gramSchmidt.InSitu{q, r})
    b.MdotM(r, q)
    if Mnorm(a.MsubM(a, b)).GetValue() < 1e-12 {
      break
    }
    a, b = b, a
  }

  return q, r, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  if n != m {
    return nil, nil, fmt.Errorf("`a' must be a square matrix")
  }
  inSitu  := &InSitu{}
  epsilon := 1e-18
  shift   := true

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case Epsilon:
      epsilon = tmp.Value
    case Shift:
      shift = tmp.Value
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
  }
  if inSitu.U == nil {
    inSitu.U = IdentityMatrix(t, n)
    inSitu.Hessenberg.V = inSitu.U
  } else {
    if n1, m1 := inSitu.U.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
    if inSitu.InitializeU {
      inSitu.U.SetIdentity()
    }
  }
  if inSitu.C == nil {
    inSitu.C = NullScalar(t)
  }
  if inSitu.S == nil {
    inSitu.S = NullScalar(t)
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
  return hessenbergQrAlgorithm(inSitu, epsilon, shift)
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
  h, u, err := Run(a, args...)
  if err != nil {
    return nil, nil, err
  }
  return h.Diag(), u, nil
}
