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

package determinant

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestDeterminant1(t *testing.T) {

	m := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})

	if r, _ := Run(m); r.GetValue() != -2 {
		t.Error("Matrix determinant failed!")
	}

}

func TestDeterminant2(t *testing.T) {

	m := NewMatrix(RealType, 3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})

	if r, _ := Run(m); r.GetValue() != 0 {
		t.Error("Matrix determinant failed!")
	}

}

func TestDeterminant3(t *testing.T) {

	m := NewMatrix(RealType, 4, 4, []float64{3, 2, 0, 1, 4, 0, 1, 2, 3, 0, 2, 1, 9, 2, 3, 1})

	if r, _ := Run(m); r.GetValue() != 24 {
		t.Error("Matrix determinant failed!")
	}

}

func TestDeterminant4(t *testing.T) {

	m := NewMatrix(RealType, 3, 3, []float64{2, -1, 0, -1, 2, -1, 0, -1, 2})

	r1, _ := Run(m)
	r2, _ := Run(m, PositiveDefinite{true})
	r3, _ := Run(m, PositiveDefinite{true}, LogScale{true})

	if math.Abs(r1.GetValue()-r2.GetValue()) > 1e-4 {
		t.Error("Matrix determinant failed!")
	}
	if math.Abs(r1.Log(r1).GetValue()-r3.GetValue()) > 1e-4 {
		t.Error("Matrix determinant failed!")
	}

}
