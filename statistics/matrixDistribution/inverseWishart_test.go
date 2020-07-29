/* Copyright (C) 2016-2020 Philipp Benner
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

package matrixDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestInverseWishartDistribution1(t *testing.T) {

  x  := NewDenseFloat64Matrix([]float64{2, -0.3, -0.3, 4}, 2, 2)
  nu := NewFloat64(3.0)
  s  := NewDenseFloat64Matrix([]float64{1, +0.3, +0.3, 1}, 2, 2)
  r  := NewFloat64(0.0)

  wishart, _ := NewInverseWishartDistribution(nu, s)
  wishart.LogPdf(r, x)

  if math.Abs(r.GetFloat64() - -9.25614036) > 1e-4 {
    t.Error("Inverse Wishart LogPdf failed!")
  }
}
