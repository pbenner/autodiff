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

package lineSearch

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestLineSearch(t *testing.T) {

	f := func(x Scalar) Scalar {
		a := Sub(x, NewReal(3.0))
		b := Pow(x, NewReal(3.0))
		c := Pow(Sub(x, NewReal(6.0)), NewReal(4.0))
		return Mul(Mul(a, b), c)
	}
	g := func(alpha Scalar) (Scalar, error) {
		return f(Add(NewReal(1.7), alpha)), nil
	}
	// hook := func(x, y, g Scalar) bool {
	//   fmt.Println("x:", x)
	//   fmt.Println("y:", y)
	//   return false
	// }
	x, err := Run(g, RealType)

	if err != nil {
		t.Error(err)
	} else {
		if math.Abs(x.GetValue()-4.381409e-02) > 1e-6 {
			t.Error("TestLineSearch failed")
		}
	}
}
