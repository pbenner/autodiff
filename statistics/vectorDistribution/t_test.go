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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestTDistribution1(t *testing.T) {

	nu := NewReal(1.0)
	mu := NewVector(RealType, []float64{2, 3})
	sigma := NewMatrix(RealType, 2, 2, []float64{2, 1, 1, 2})

	distribution, _ := NewTDistribution(nu, mu, sigma)

	x := NewVector(RealType, []float64{1, 2})
	y := NewReal(0.0)

	distribution.LogPdf(y, x)

	if math.Abs(y.GetValue() - -3.153422e+00) > 1e-4 {
		t.Error("T LogPdf failed!")
	}
}
