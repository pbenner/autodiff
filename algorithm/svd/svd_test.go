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

package svd

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "sort"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {
	a := NewMatrix(RealType, 5, 5, []float64{
		1, 1, 0, 0, 0,
		0, 2, 1, 0, 0,
		0, 0, 3, 1, 0,
		0, 0, 0, 4, 1,
		0, 0, 0, 0, 5})

	h, _, _, _ := Run(a)

	r := []float64{}
	for i := 0; i < 4; i++ {
		r = append(r, h.At(i, i).GetValue())
	}
	sort.Float64s(r)

	singularValues := []float64{8.584972e-01, 2.110322e+00, 3.086113e+00, 4.086174e+00, 5.252486e+00}

	for i := 0; i < len(r); i++ {
		if math.Abs(r[i]-singularValues[i]) > 1e-5 {
			t.Errorf("test failed for singular value `%d'", i)
		}
	}
}

func Test2(t *testing.T) {
	a := NewMatrix(RealType, 3, 3, []float64{
		1, 1, 0,
		0, 2, 1,
		0, 0, 3})

	h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

	d := MdotM(MdotM(u.T(), a), v)

	if Mnorm(MsubM(d, h)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func Test3(t *testing.T) {
	a := NewMatrix(RealType, 7, 5, []float64{
		1, 1, 0, 0, 0,
		0, 2, 1, 0, 0,
		0, 0, 3, 1, 0,
		0, 0, 0, 4, 1,
		0, 0, 0, 0, 5,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0})

	h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

	d := MdotM(MdotM(u.T(), a), v)
	b := MdotM(MdotM(u, h), v.T())

	if Mnorm(MsubM(d, h)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
	if Mnorm(MsubM(a, b)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func Test4(t *testing.T) {
	a := NewMatrix(RealType, 7, 7, []float64{
		1, 1, 0, 0, 0, 0, 0,
		0, 2, 1, 0, 0, 0, 0,
		0, 0, 3, 1, 0, 0, 0,
		0, 0, 0, 4, 1, 0, 0,
		0, 0, 0, 0, 1e-21, 1, 0,
		0, 0, 0, 0, 0, 6, 1,
		0, 0, 0, 0, 0, 0, 7})

	h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

	d := MdotM(MdotM(u.T(), a), v)
	b := MdotM(MdotM(u, h), v.T())

	if Mnorm(MsubM(d, h)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
	if Mnorm(MsubM(a, b)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func Test5(t *testing.T) {
	a := NewMatrix(RealType, 7, 7, []float64{
		1, 1, 0, 0, 0, 0, 0,
		0, 2, 1, 0, 0, 0, 0,
		0, 0, 3, 1, 0, 0, 0,
		0, 0, 0, 4, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 6, 1,
		0, 0, 0, 0, 0, 0, 7})

	h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

	d := MdotM(MdotM(u.T(), a), v)
	b := MdotM(MdotM(u, h), v.T())

	if Mnorm(MsubM(d, h)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
	if Mnorm(MsubM(a, b)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func Test6(t *testing.T) {
	a := NewMatrix(RealType, 16, 16, []float64{
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1})

	p, q := splitMatrix(a, 0)

	if p != 6 {
		t.Error("test failed")
	}
	if q != 3 {
		t.Error("test failed")
	}
}
