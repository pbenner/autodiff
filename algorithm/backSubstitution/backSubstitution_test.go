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

package backSubstitution

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {
	a := NewMatrix(RealType, 3, 3, []float64{
		1, -2, 1,
		0, 1, 6,
		0, 0, 1})
	b := NewVector(RealType, []float64{
		4, -1, 2})
	r := NewVector(RealType, []float64{
		-24, -13, 2})

	x, _ := Run(a, b)

	if Vnorm(VsubV(r, x)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}
