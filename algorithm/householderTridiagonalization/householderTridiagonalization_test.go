/* Copyright (C) 2017 Philipp Benner
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
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {
	a := NewMatrix(RealType, 4, 4, []float64{
		4, 2, 2, 1,
		2, -3, 1, 1,
		2, 1, 3, 1,
		1, 1, 1, 2})

	b, u, _ := Run(a, ComputeU{true})

	r1 := NewMatrix(RealType, 4, 4, []float64{
		4.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00,
		3.000000e+00, 2.000000e+00, 3.162278e+00, 0.000000e+00,
		0.000000e+00, 3.162278e+00, -1.400000e+00, 2.000000e-01,
		0.000000e+00, 0.000000e+00, 2.000000e-01, 1.400000e+00})
	r2 := MdotM(u.T(), MdotM(a, u))

	if Mnorm(MsubM(r1, b)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
	if Mnorm(MsubM(r2, b)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}
