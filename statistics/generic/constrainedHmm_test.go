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

package generic

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestChmm1(t *testing.T) {
	tr := NewMatrix(BareRealType, 4, 4, []float64{
		1, 2, 0, 4,
		5, 6, 7, 8,
		0, 4, 1, 2,
		7, 8, 5, 6})

	x := NullVector(RealType, 4)
	l := NullVector(RealType, 4)
	l.At(0).SetValue(math.Log(4.336))
	l.At(1).SetValue(math.Log(28.664))
	l.At(2).SetValue(math.Log(4.336))
	l.At(3).SetValue(math.Log(28.664))

	c1, _ := NewEqualityConstraint([]int{
		0, 3,
		1, 2,
		1, 3})
	c2, _ := NewEqualityConstraint([]int{
		2, 1,
		3, 0,
		3, 1})

	r, err := newChmmTransitionMatrix(tr, []EqualityConstraint{c1, c2}, false, false)
	if err != nil {
		t.Error(err)
		return
	}
	r.EvalConstraints(l, x)

	if Vnorm(x).GetValue() > 1e-4 {
		t.Error("test failed")
	}
}

func TestChmm2(t *testing.T) {
	tr := NewMatrix(BareRealType, 4, 4, []float64{
		1, 2, 0, 4,
		5, 6, 7, 8,
		0, 4, 1, 2,
		7, 8, 5, 6})

	l1 := 4.336021
	l2 := 28.663970

	sr := NewMatrix(RealType, 4, 4, []float64{
		1 / l1, 2 / l1, 0, 19 / (1*l1 + 2*l2),
		5 / l2, 6 / l2, 19 / (1*l1 + 2*l2), 19 / (1*l1 + 2*l2),
		0, 19 / (1*l1 + 2*l2), 1 / l1, 2 / l1,
		19 / (1*l1 + 2*l2), 19 / (1*l1 + 2*l2), 5 / l2, 6 / l2})
	sr.Map(func(a Scalar) { a.Log(a) })

	c1, _ := NewEqualityConstraint([]int{
		0, 3,
		1, 2,
		1, 3})
	c2, _ := NewEqualityConstraint([]int{
		2, 1,
		3, 0,
		3, 1})

	if r, err := NewChmmTransitionMatrix(tr, []EqualityConstraint{c1, c2}, false); err != nil {
		t.Error("test failed:", err)
	} else {
		if Mnorm(MsubM(r, sr)).GetValue() > 1e-4 {
			t.Error("test failed")
		}
	}
}
