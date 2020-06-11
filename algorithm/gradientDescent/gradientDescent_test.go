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

package gradientDescent

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
	m1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	m2 := m1.CloneMatrix()
	m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

	rows, cols := m1.Dims()
	if rows != cols {
		panic("MInverse(): Not a square matrix!")
	}
	I := IdentityMatrix(m1.ElementType(), rows)
	// objective function
	f := func(x Vector) (Scalar, error) {
		m2.AsVector().Set(x)
		s := Mnorm(MsubM(MdotM(m1, m2), I))
		return s, nil
	}
	x, _ := Run(f, m2.AsVector(), 0.01)
	m2.AsVector().Set(x)

	if Mnorm(MsubM(m2, m3)).GetValue() > 1e-8 {
		t.Error("Inverting matrix failed!")
	}
}
