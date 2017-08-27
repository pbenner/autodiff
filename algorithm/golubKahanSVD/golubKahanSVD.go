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

package golubKahanSVD

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/givensRotation"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type InSitu struct {
  T1 Scalar
  T2 Scalar
  T3 Scalar
  T4 Scalar
  L1 Scalar
  L2 Scalar
  C2 Scalar
  C4 Scalar
}

/* -------------------------------------------------------------------------- */

// compute eigenvalues of a symmetric
// 2x2 matrix [ t1 t3; t3 t2 ] and store
// the result in l1 and l2
func eigenvalues(t1, t2, t3, l1, l2, t, c2, c4 Scalar) {
  t.Sub(l1, l2)
  t.Mul(t, t)

  l1.Mul(t3, t3)
  l1.Mul(c4, l1)

  t.Add(t, l1)

  l2.Add(t1, t2)
  l1.Add(l2, t)
  l2.Sub(l2, t)

  l1.Div(l1, c2)
  l2.Div(l2, c2)
}

/* -------------------------------------------------------------------------- */

func golubKahanSVDstep(B Matrix, inSitu *InSitu, epsilon float64) (Matrix, error) {

  var mu Scalar

  m, n := B.Dims()

  c2 := inSitu.C2
  c4 := inSitu.C4

  // compute the eigenvalues of the trailing 2-by-2 submatrix of T = B^t B
  l1 := inSitu.L1
  l2 := inSitu.L2
  t1 := inSitu.T1
  t2 := inSitu.T2
  t3 := inSitu.T3
  t  := inSitu.T4

  t1.SetValue(0.0)
  t2.SetValue(0.0)
  t3.SetValue(0.0)
  // compute:
  // trailing 2-by-2 submatrix of T = B^t B: [ t1, t3; t3, t2 ]
  for i := 0; i < m; i++ {
    t.Mul(B.At(n-2, i), B.At(n-2, i))
    t1.Add(t1, t)
    t.Mul(B.At(n-1, i), B.At(n-1, i))
    t2.Add(t2, t)
    t.Mul(B.At(n-1, i), B.At(n-2, i))
    t3.Add(t3, t)
  }
  eigenvalues(t1, t2, t3, l1, l2, t, c2, c4)

  if math.Abs(l1.GetValue() - t2.GetValue()) < math.Abs(l2.GetValue() - t2.GetValue()) {
    // l1 is closer to t2
    mu = l1
  } else {
    // l2 is closer to t2
    mu = l2
  }
  y := t1
  y.Sub(y, mu)
  z := t3
  c := t2
  s := t

  for k := 0; k < n-1; k++ {
    givensRotation.Run(y, z, c, s)
    givensRotation.RunApplyRight(B, c, s, k, k+1, t1, t3)
    y.Set(B.At(k+0, k))
    z.Set(B.At(k+1, k))
    givensRotation.Run(y, z, c, s)
    givensRotation.RunApplyLeft(B, c, s, k, k+1, t1, t3)
    if k < n-2 {
      y.Set(B.At(k,k+1))
      z.Set(B.At(k,k+2))
    }
  }
  return nil, nil
}

/* -------------------------------------------------------------------------- */

func golubKahanSVD(inSitu *InSitu, epsilon float64) (Matrix, error) {

  
  return nil, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, error) {

  m, n := a.Dims()
//  t    := a.ElementType()

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
  return golubKahanSVD(inSitu, epsilon)
}
