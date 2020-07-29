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

package eigensystem

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
     7,  3,  4, -11, -9, -2,
    -6,  4, -5,   7,  1, 12,
    -1, -9,  2,   2,  9,  1,
    -8,  0, -1,   5,  0,  8,
    -4,  3, -5,   7,  2, 10,
    6,  1,  4, -11, -7, -1}, 6, 6)

  eigenvalues := []float64{5, 5, 4, 3, 1, 1}

  if r, v, err := Run(a, ComputeEigenvectors{false}); err != nil {
    test.Error("test failed")
  } else {
    for i := 0; i < 4; i++ {
      if math.Abs(r.At(i).GetFloat64()-eigenvalues[i]) > 1e-5 {
        test.Errorf("test failed for eigenvalue `%d'", i)
      }
    }
    if v != nil {
      test.Error("test failed")
    }
  }
}

func Test2(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 2,  3, 4,
    4, 4,  4, 4,
    0, 1, -1, 1,
    0, 0,  2, 3 }, 4, 4)
  t := NewFloat64(0.0)

  eigenvalues  := NewDenseFloat64Vector([]float64{
    6.741657e+00, 2.561553e+00, -1.561553e+00, -7.416574e-01})
  eigenvectors := NewDenseFloat64Matrix([]float64{
    4.229518e-01, -6.818712e-02,  3.347805e-01, -7.125998e-01,
    8.951414e-01, -8.649304e-01,  1.055702e-01,  4.020316e-01,
    1.242021e-01, -1.064778e-01, -8.575579e-01,  5.070618e-01,
    6.638882e-02,  4.857041e-01,  3.759939e-01, -2.710359e-01 }, 4, 4)

  if e, v, err := Run(a); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(e.VsubV(eigenvalues, e)).GetFloat64() > 1e-4 {
      test.Error("test failed")
    }
    if t.Mnorm(v.MsubM(eigenvectors, v)).GetFloat64() > 1e-4 {
      test.Error("test failed")
    }
  }
}
