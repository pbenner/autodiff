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

package matrixDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "os"
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestShapeHmm1(t *testing.T) {

	// ShapeHmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))

	d1, _ := vectorDistribution.NewScalarId(c1, c1, c1, c1, c1)
	d2, _ := vectorDistribution.NewScalarId(c2, c2, c2, c2, c2)

	e1, _ := NewVectorId(d1)
	e2, _ := NewVectorId(d2)

	hmm1, err := NewShapeHmm(pi, tr, nil, []MatrixPdf{e1, e2})
	if err != nil {
		t.Error(err)
	}
	// test export/import
	//////////////////////////////////////////////////////////////////////////////
	hmm2 := &ShapeHmm{}

	filename := "shapeHmm_test.json"

	if err := ExportDistribution(filename, hmm1); err != nil {
		t.Error(err)
		return
	}
	if r, err := ImportMatrixPdf(filename, BareRealType); err != nil {
		t.Error(err)
		return
	} else {
		hmm2 = r.(*ShapeHmm)
	}
	p1 := hmm1.GetParameters()
	p2 := hmm2.GetParameters()

	if Vnorm(VsubV(p1, p2)).GetValue() > 1e-6 {
		t.Error("test failed")
	}
	os.Remove(filename)
}
