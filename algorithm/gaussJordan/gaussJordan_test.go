/* Copyright (C) 2015-2020 Philipp Benner
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

package gaussJordan

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestGaussJordan1(test *testing.T) {
  n := 3
  a := NewDenseFloat64Matrix([]float64{1, 1, 1, 2, 1, 1, 1, 2, 1}, n, n)
  b := NewDenseFloat64Vector([]float64{1,1,1})
  r := NewDenseFloat64Matrix([]float64{-1, 1, 0, -1, 0, 1, 3, -1, -1}, n, n)
  t := NewFloat64(0.0)
  x := NullDenseFloat64Matrix(n, n)
  x.SetIdentity()

  if err := Run(a, x, b); err != nil {
    test.Error(err)
  } else {
    if t.Mnorm(r.MsubM(x, r)).GetFloat64() > 1e-4 {
      test.Error("Gauss-Jordan method failed!")
    }
  }
}

func TestGaussJordan2(test *testing.T) {
  n := 5
  a := NewDenseFloat64Matrix([]float64{
    2, 7, 1, 8, 2,
    0, 8, 1, 8, 2,
    0, 0, 8, 4, 5,
    0, 0, 0, 9, 0,
    0, 0, 0, 0, 4 }, n, n)
  b := NewDenseFloat64Vector([]float64{1,1,1,1,1})
  r := NewDenseFloat64Matrix([]float64{
    5.000000e-01, -4.375000e-01, -7.812500e-03, -5.208333e-02, -2.148438e-02,
    0.000000e+00,  1.250000e-01, -1.562500e-02, -1.041667e-01, -4.296875e-02,
    0.000000e+00,  0.000000e+00,  1.250000e-01, -5.555556e-02, -1.562500e-01,
    0.000000e+00,  0.000000e+00,  0.000000e+00,  1.111111e-01,  0.000000e+00,
    0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  2.500000e-01 }, n, n)
  t := NewFloat64(0.0)
  x := NullDenseFloat64Matrix(n, n)
  x.SetIdentity()

  if err := Run(a, x, b, UpperTriangular{true}); err != nil {
    test.Error(err)
  } else {
    if t.Mnorm(r.MsubM(x, r)).GetFloat64() > 1e-4 {
      test.Error("Gauss-Jordan method failed!")
    }
  }
}
