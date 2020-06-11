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

package gaussJordan

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestGaussJordan1(t *testing.T) {
	n := 3
	a := NewMatrix(RealType, n, n, []float64{1, 1, 1, 2, 1, 1, 1, 2, 1})
	x := IdentityMatrix(RealType, n)
	b := NewVector(RealType, []float64{1, 1, 1})
	r := NewMatrix(RealType, n, n, []float64{-1, 1, 0, -1, 0, 1, 3, -1, -1})

	if err := Run(a, x, b); err != nil {
		t.Error(err)
	} else {
		if Mnorm(MsubM(x, r)).GetValue() > 1e-4 {
			t.Error("Gauss-Jordan method failed!")
		}
	}
}

func TestGaussJordan2(t *testing.T) {
	n := 5
	a := NewMatrix(RealType, n, n, []float64{
		2, 7, 1, 8, 2,
		0, 8, 1, 8, 2,
		0, 0, 8, 4, 5,
		0, 0, 0, 9, 0,
		0, 0, 0, 0, 4})
	x := IdentityMatrix(RealType, n)
	b := NewVector(RealType, []float64{1, 1, 1, 1, 1})
	r := NewMatrix(RealType, n, n, []float64{
		5.000000e-01, -4.375000e-01, -7.812500e-03, -5.208333e-02, -2.148438e-02,
		0.000000e+00, 1.250000e-01, -1.562500e-02, -1.041667e-01, -4.296875e-02,
		0.000000e+00, 0.000000e+00, 1.250000e-01, -5.555556e-02, -1.562500e-01,
		0.000000e+00, 0.000000e+00, 0.000000e+00, 1.111111e-01, 0.000000e+00,
		0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.500000e-01})

	if err := Run(a, x, b, UpperTriangular{true}); err != nil {
		t.Error(err)
	} else {
		if Mnorm(MsubM(x, r)).GetValue() > 1e-4 {
			t.Error("Gauss-Jordan method failed!")
		}
	}
}
