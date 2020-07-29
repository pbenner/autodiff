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

package householderBidiagonalization

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
     1,  2,  3,
     4,  5,  6,
     7,  8,  9,
    10, 11, 12}, 4, 3)

  b, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

  r1 := NewDenseFloat64Matrix([]float64{
    1.288410e+01, 2.187643e+01,  0.000000e+00,
    0.000000e+00, 2.246235e+00, -6.132813e-01,
    0.000000e+00, 0.000000e+00,  0.000000e+00,
    0.000000e+00, 0.000000e+00,  0.000000e+00}, 4, 3)
  r2 := NullDenseFloat64Matrix(4, 3)
  r2.MdotM(a,v)
  r2.MdotM(u.T(), r2)
  t  := NewFloat64(0.0)

  if t.Mnorm(r1.MsubM(r1, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
  if t.Mnorm(r1.MsubM(r2, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}
