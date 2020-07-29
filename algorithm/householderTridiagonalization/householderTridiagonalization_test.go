/* Copyright (C) 2017-2020 Philipp Benner
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

package householderTridiagonalization

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    4,  2,  2, 1,
    2, -3,  1, 1,
    2,  1,  3, 1,
    1,  1,  1, 2 }, 4, 4)

  b, u, _ := Run(a, ComputeU{true})

  t  := NewFloat64(0.0)
  r1 := NewDenseFloat64Matrix([]float64{
    4.000000e+00, 3.000000e+00,  0.000000e+00, 0.000000e+00,
    3.000000e+00, 2.000000e+00,  3.162278e+00, 0.000000e+00,
    0.000000e+00, 3.162278e+00, -1.400000e+00, 2.000000e-01,
    0.000000e+00, 0.000000e+00,  2.000000e-01, 1.400000e+00}, 4, 4)
  r2 := NullDenseFloat64Matrix(4, 4)
  r2.MdotM(a,u)
  r2.MdotM(u.T(), r2)

  if t.Mnorm(r1.MsubM(r1, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
  if t.Mnorm(r2.MsubM(r2, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}
