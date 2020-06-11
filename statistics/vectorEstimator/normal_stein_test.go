/* Copyright (C) 2019 Philipp Benner
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

package vectorEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "testing"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/eigensystem"

/* -------------------------------------------------------------------------- */

func TestNormalStein1(t *testing.T) {
	si := NewMatrix(BareRealType, 3, 3, []float64{
		66, 78, 90,
		78, 93, 108,
		90, 108, 126})
	l, _, err := eigensystem.Run(si, eigensystem.Symmetric{true})
	if err != nil {
		t.Error("test failed")
		return
	}
	v := NullReal()
	s := NewVector(BareRealType, []float64{2.363898e+02, 1.142335e+00, -4.361160e-14})
	r := (NormalSteinEstimator{}).steinEigen(l, 10)

	if v.Vnorm(s.VsubV(s, r)); v.GetValue() > 1e-4 {
		t.Error("test failed")
	}
}

func TestNormalStein2(t *testing.T) {
	si := NewMatrix(BareRealType, 3, 3, []float64{
		66, 78, 90,
		78, 93, 108,
		90, 108, 126})
	l, H, err := eigensystem.Run(si, eigensystem.Symmetric{true})
	if err != nil {
		t.Error("test failed")
		return
	}
	l = (NormalSteinEstimator{}).steinEigen(l, 10)

	// compute covariance matrix using l and H
	for i := 0; i < l.Dim(); i++ {
		for j := 0; j < l.Dim(); j++ {
			si.At(i, j).Mul(H.ConstAt(j, i), l.ConstAt(i))
		}
	}
	si.MdotM(H, si)

	v := NullReal()
	s := NewMatrix(BareRealType, 3, 3, []float64{
		55.07872, 64.96757, 74.85642,
		64.96757, 77.44899, 89.93040,
		74.85642, 89.93040, 105.00439})

	if v.Mnorm(si.MsubM(si, s)); v.GetValue() > 1e-4 {
		t.Error("test failed")
	}
}
