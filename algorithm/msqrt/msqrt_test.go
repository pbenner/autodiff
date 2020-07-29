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

package msqrt

/* -------------------------------------------------------------------------- */

import   "testing"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestMSqrt(test *testing.T) {
  n := 2
  a := NewDenseFloat64Matrix([]float64{2, 1, 1, 2}, n, n)
  x, _ := Run(a)
  r := NewDenseFloat64Matrix([]float64{1.366025e+00, 3.660254e-01, 3.660254e-01, 1.366025e+00}, n, n)
  t := NewFloat64(0.0)

  if t.Mnorm(x.MsubM(x, r)).GetFloat64() > 1e-8 {
    test.Error("MSqrt failed!")
  }
}
