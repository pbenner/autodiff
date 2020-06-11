/* Copyright (C) 2016 Philipp Benner
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
import "math"
import "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestInverseWishartDistribution1(t *testing.T) {

	x := NewMatrix(RealType, 2, 2, []float64{2, -0.3, -0.3, 4})
	nu := NewReal(3.0)
	s := NewMatrix(RealType, 2, 2, []float64{1, +0.3, +0.3, 1})
	r := NewReal(0.0)

	wishart, _ := NewInverseWishartDistribution(nu, s)
	wishart.LogPdf(r, x)

	if math.Abs(r.GetValue() - -9.25614036) > 1e-4 {
		t.Error("Inverse Wishart LogPdf failed!")
	}
}
