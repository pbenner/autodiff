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
//import   "math"
import "os"
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import "github.com/pbenner/autodiff/statistics/generic"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestHhmm1(t *testing.T) {
	// HMM definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{1, 1, 1, 1})

	tr := NewMatrix(RealType, 4, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16})

	sr := NewMatrix(RealType, 4, 4, []float64{
		1.296296e-01, 2.592593e-01, 3.055556e-01, 3.055556e-01,
		1.767677e-01, 2.121212e-01, 3.055556e-01, 3.055556e-01,
		2.300000e-01, 2.300000e-01, 2.582609e-01, 2.817391e-01,
		2.300000e-01, 2.300000e-01, 2.612903e-01, 2.787097e-01})
	sr.Map(func(a Scalar) { a.Log(a) })

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	c3, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c4, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2, c3, c4}

	tree := generic.NewHmmNode(generic.NewHmmLeaf(0, 2), generic.NewHmmLeaf(2, 4))

	hmm, err := NewHierarchicalHmm(pi, tr, nil, edist, tree)
	if err != nil {
		t.Error(err)
	}

	if Mnorm(MsubM(hmm.Tr, sr)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func TestHhmm2(t *testing.T) {
	// HMM definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1})

	tr := NewMatrix(RealType, 6, 6, []float64{
		1, 2, 3, 4, 1, 2,
		5, 6, 7, 8, 3, 4,
		9, 10, 11, 12, 5, 6,
		13, 14, 15, 16, 7, 8,
		9, 10, 11, 12, 13, 14,
		15, 16, 17, 18, 19, 20})

	sr := NewMatrix(RealType, 6, 6, []float64{
		1.024978e-01, 2.049957e-01, 2.416021e-01, 2.416021e-01, 1.046512e-01, 1.046512e-01,
		1.397698e-01, 1.677237e-01, 2.416021e-01, 2.416021e-01, 1.046512e-01, 1.046512e-01,
		1.818605e-01, 1.818605e-01, 2.042063e-01, 2.227705e-01, 1.046512e-01, 1.046512e-01,
		1.818605e-01, 1.818605e-01, 2.066017e-01, 2.203751e-01, 1.046512e-01, 1.046512e-01,
		1.551724e-01, 1.551724e-01, 1.551724e-01, 1.551724e-01, 1.826309e-01, 1.966794e-01,
		1.551724e-01, 1.551724e-01, 1.551724e-01, 1.551724e-01, 1.847922e-01, 1.945181e-01})
	sr.Map(func(a Scalar) { a.Log(a) })

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	c3, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c4, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	c5, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c6, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))

	edist := []ScalarPdf{c1, c2, c3, c4, c5, c6}

	tree := generic.NewHmmNode(
		generic.NewHmmNode(generic.NewHmmLeaf(0, 2), generic.NewHmmLeaf(2, 4)),
		generic.NewHmmLeaf(4, 6))

	hmm, err := NewHierarchicalHmm(pi, tr, nil, edist, tree)
	if err != nil {
		t.Error(err)
	}

	if Mnorm(MsubM(hmm.Tr, sr)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
}

func TestHhmm3(t *testing.T) {
	filename := "hierarchicalHmm_test.json"
	// HMM definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{1, 1, 1, 1})

	tr := NewMatrix(RealType, 4, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16})

	sr := NewMatrix(RealType, 4, 4, []float64{
		1.296296e-01, 2.592593e-01, 3.055556e-01, 3.055556e-01,
		1.767677e-01, 2.121212e-01, 3.055556e-01, 3.055556e-01,
		2.300000e-01, 2.300000e-01, 2.582609e-01, 2.817391e-01,
		2.300000e-01, 2.300000e-01, 2.612903e-01, 2.787097e-01})
	sr.Map(func(a Scalar) { a.Log(a) })

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	c3, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c4, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2, c3, c4}

	tree := generic.NewHmmNode(generic.NewHmmLeaf(0, 2), generic.NewHmmLeaf(2, 4))

	hhmm1, err := NewHierarchicalHmm(pi, tr, nil, edist, tree)
	if err != nil {
		t.Error(err)
		return
	}

	if err := ExportDistribution(filename, hhmm1); err != nil {
		t.Error(err)
		return
	}
	if hhmm2, err := ImportVectorPdf(filename, BareRealType); err != nil {
		t.Error(err)
		return
	} else {
		t1 := hhmm1.Tr
		t1.Map(func(x Scalar) { x.Exp(x) })
		t2 := hhmm2.(*Hhmm).Tr
		t2.Map(func(x Scalar) { x.Exp(x) })
		if Mnorm(MsubM(t1, t2)).GetValue() > 1e-8 {
			t.Error("test failed")
			return
		}
		tree1 := hhmm1.Tr.(generic.HhmmTransitionMatrix).Tree
		tree2 := hhmm2.(*Hhmm).Tr.(generic.HhmmTransitionMatrix).Tree
		if len(tree1.Children) != len(tree2.Children) {
			t.Error("test failed")
			return
		}
	}
	os.Remove(filename)
}
