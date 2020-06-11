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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "os"
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import "github.com/pbenner/autodiff/statistics/generic"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestChmm1(t *testing.T) {
	filename := "constrainedHmm_test.json"

	pi := NewVector(RealType, []float64{1, 1, 1, 1})

	tr := NewMatrix(BareRealType, 4, 4, []float64{
		1, 2, 0, 4,
		5, 6, 7, 8,
		0, 4, 1, 2,
		7, 8, 5, 6})

	const1, _ := generic.NewEqualityConstraint([]int{
		0, 3,
		1, 2,
		1, 3})
	const2, _ := generic.NewEqualityConstraint([]int{
		2, 1,
		3, 0,
		3, 1})

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	c3, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c4, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2, c3, c4}

	chmm1, err := NewConstrainedHmm(pi, tr, nil, edist, []generic.EqualityConstraint{const1, const2})
	if err != nil {
		t.Error("test failed:", err)
		return
	}

	if err := ExportDistribution(filename, chmm1); err != nil {
		t.Error(err)
		return
	}
	if chmm2, err := ImportVectorPdf(filename, BareRealType); err != nil {
		t.Error(err)
		return
	} else {
		t1 := chmm1.Tr
		t1.Map(func(x Scalar) { x.Exp(x) })
		t2 := chmm2.(*Chmm).Tr
		t2.Map(func(x Scalar) { x.Exp(x) })
		if Mnorm(MsubM(t1, t2)).GetValue() > 1e-8 {
			t.Error("test failed")
			return
		}
		c1 := chmm1.Tr.(generic.ChmmTransitionMatrix).GetConstraints()
		c2 := chmm2.(*Chmm).Tr.(generic.ChmmTransitionMatrix).GetConstraints()
		if len(c1) != len(c2) {
			t.Error("test failed")
			return
		}
		for i := 0; i < len(c1); i++ {
			if len(c1[i]) != len(c2[i]) {
				t.Error("test failed")
				return
			}
			for j := 0; j < len(c1[i]); j++ {
				if c1[i][j][0] != c1[i][j][0] {
					t.Error("test failed")
					return
				}
				if c1[i][j][1] != c1[i][j][1] {
					t.Error("test failed")
					return
				}
			}
		}
	}
	os.Remove(filename)
}
