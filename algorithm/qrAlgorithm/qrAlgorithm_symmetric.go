/* Copyright (C) 2015-2017 Philipp Benner
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

//import   "fmt"
//import   "math"

import . "github.com/pbenner/autodiff"
//import   "github.com/pbenner/autodiff/algorithm/givensRotation"
import   "github.com/pbenner/autodiff/algorithm/householderTridiagonalization"

/* -------------------------------------------------------------------------- */

// compute the eigenvalue of a symmetric
// 2x2 matrix [ t11 t12; t12 t22 ] closer
// to t22
func wilkinsonShift(mu, t11, t12, t22, c2, t1, t2 Scalar) {
  d := t1
  t := t2
  d.Sub(t11, t22)
  d.Div(d, c2)     // d = (t11 - t22)/2

  t .Mul(t12, t12)
  mu.Mul(d, d)
  mu.Add(mu, t)    // mu = d^2 + t12^2
  mu.Sqrt(mu)      // mu = sqrt(d^2 + t12^2)

  if d.GetValue() < 0.0 {
    mu.Neg(mu)
  }
  mu.Add(d, mu)    // mu = d + sign(d) sqrt(d^2 + t12^2)
  mu.Div(t, mu)    // mu = t12^2 / (d + sign(d) sqrt(d^2 + t12^2))

  mu.Sub(t22, mu)  // mu = t22 - t12^2 / (d + sign(d) sqrt(d^2 + t12^2))
}

/* -------------------------------------------------------------------------- */

func symmetricQRstep(T, Z Matrix, p, q int, inSitu *InSitu) {
}

/* -------------------------------------------------------------------------- */

func qrAlgorithmSymmetric(inSitu *InSitu, epsilon float64) (Matrix, Matrix, error) {

  T    := inSitu.H
  Z    := inSitu.U
  n, _ := T.Dims()

  if T_, Z_, err := householderTridiagonalization.Run(T, &inSitu.Householder, householderTridiagonalization.ComputeU{Z != nil}); err != nil {
    return nil, nil, err
  } else {
    T = T_
    Z = Z_
  }

  for p, q := 0, 0; q < n; {
    // p: number of rows/cols in H11
    // q: number of rows/cols in H33
    p, q = splitMatrix(T, q)

    if q < n {
      symmetricQRstep(T, Z, p, q, inSitu)
    }
  }
  return T, Z, nil
}
