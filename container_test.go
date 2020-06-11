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

package autodiff

/* -------------------------------------------------------------------------- */

import "testing"

/* -------------------------------------------------------------------------- */

func TestContainer(t *testing.T) {

	v := NewVector(RealType, []float64{1, 2, 3, 4})
	m := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})

	// test if container interface is implements
	var c1 ScalarContainer
	var c2 ScalarContainer

	c1 = v
	c2 = m

	c1.Map(func(x Scalar) { x.Mul(x, x) })
	c2.Map(func(x Scalar) { x.Mul(x, x) })

	if v.At(1).GetValue() != 4.0 {
		t.Error("Vector initialization failed!")
	}
	if m.At(0, 1).GetValue() != 4.0 {
		t.Error("Vector initialization failed!")
	}

}
