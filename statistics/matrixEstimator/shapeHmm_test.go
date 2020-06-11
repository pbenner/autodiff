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

package matrixEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarEstimator"
import "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func TestShapeHmm1(t *testing.T) {

	// ShapeHmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.1, 0.9})
	c2, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.7, 0.3})

	d1, _ := vectorEstimator.NewScalarBatchId(c1)
	d2, _ := vectorEstimator.NewScalarBatchId(c2)

	e1, _ := NewVectorBatchId(d1, d1, d1, d1, d1)
	e2, _ := NewVectorBatchId(d2, d2, d2, d2, d2)

	if estimator, err := NewShapeHmmEstimator(pi, tr, nil, []MatrixBatchEstimator{e1, e2}, 1e-8, -1); err != nil {
		t.Error(err)
	} else {
		x := NewMatrix(RealType, 10, 1, []float64{
			1, 1, 1, 1, 1,
			0, 0, 0, 0, 0})

		if err := estimator.EstimateOnData([]ConstMatrix{x}, nil, ThreadPool{}); err != nil {
			t.Error(err)
			return
		}
		d, _ := estimator.GetEstimate()
		r := NewReal(0.0)

		if err := d.LogPdf(r, x); err != nil {
			t.Error(err)
			return
		}
		if math.Abs(r.GetValue() - -1.550184e+01) > 1e-4 {
			t.Errorf("test failed")
		}
	}
}

func TestShapeHmm2(t *testing.T) {

	// ShapeHmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c11, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.1, 0.9})
	c12, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.9, 0.1})
	c1, _ := vectorEstimator.NewScalarBatchId(c11, c12)

	c21, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.7, 0.3})
	c22, _ := scalarEstimator.NewCategoricalEstimator(
		[]float64{0.3, 0.7})
	c2, _ := vectorEstimator.NewScalarBatchId(c21, c22)

	d1, _ := NewVectorBatchId(c1, c1, c1, c1, c1)
	d2, _ := NewVectorBatchId(c2, c2, c2, c2, c2)

	if estimator, err := NewShapeHmmEstimator(pi, tr, nil, []MatrixBatchEstimator{d1, d2}, 1e-8, -1); err != nil {
		t.Error(err)
	} else {
		x := NewMatrix(RealType, 100, 2, []float64{
			1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
			1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
			1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
			1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0})

		if err := estimator.EstimateOnData([]ConstMatrix{x}, nil, ThreadPool{}); err != nil {
			t.Error(err)
			return
		}
		d, _ := estimator.GetEstimate()
		r := NewReal(0.0)

		if err := d.LogPdf(r, x); err != nil {
			t.Error(err)
			return
		}
		if math.Abs(r.GetValue() - -4.891673e+02) > 1e-4 {
			t.Errorf("test failed")
		}
	}
}
