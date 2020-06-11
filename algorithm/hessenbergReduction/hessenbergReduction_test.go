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

package hessenbergReduction

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test(t *testing.T) {

	a := NewMatrix(RealType, 5, 5, []float64{
		3.5, 3.0, 4.0, 32.5, 0.4,
		3.0, 8.6, 0.4, 25.4, 2.5,
		4.0, 0.4, 6.4, 38.0, 0.4,
		32.5, 25.4, 38.0, 304.0, 1.3,
		0.4, 2.5, 0.4, 1.3, 3.6})

	r := NewMatrix(RealType, 5, 5, []float64{
		3.5, 32.8848, 0.0, 0.0, 0.0,
		32.8848, 310.857, 3.2874, 0.0, 0.0,
		0.0, 3.2874, 8.03978, 1.73313, 0.0,
		0.0, 0.0, 1.73313, 3.73586, -0.400312,
		0.0, 0.0, 0.0, -0.400312, -0.0328116})

	b, u, _ := Run(a, ComputeU{true})
	// apply similarity transform
	c := MdotM(MdotM(u.T(), a), u)

	if Mnorm(MsubM(b, r)).GetValue() > 1e-4 {
		t.Error("test failed")
	}
	if Mnorm(MsubM(b, c)).GetValue() > 1e-4 {
		t.Error("test failed")
	}
}
