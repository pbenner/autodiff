/* Copyright (C) 2015 Philipp Benner
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

import "testing"
import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestMSqrt(t *testing.T) {
	n := 2
	a := NewMatrix(RealType, n, n, []float64{2, 1, 1, 2})
	x, _ := Run(a)
	r := NewMatrix(RealType, n, n, []float64{1.366025e+00, 3.660254e-01, 3.660254e-01, 1.366025e+00})

	if Mnorm(MsubM(x, r)).GetValue() > 1e-8 {
		t.Error("MSqrt failed!")
	}
}
