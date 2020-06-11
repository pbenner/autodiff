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

import "fmt"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ShapeHmmAdapter struct {
	Estimator MatrixBatchEstimator
	estimate  MatrixPdf
	x         []ConstMatrix
	n         int
}

/* -------------------------------------------------------------------------- */

func NewShapeHmmAdapter(estimator MatrixBatchEstimator) (*ShapeHmmAdapter, error) {
	r := ShapeHmmAdapter{}
	r.Estimator = estimator
	if err := r.updateEstimate(); err != nil {
		return nil, err
	}
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmmAdapter) Clone() *ShapeHmmAdapter {
	r := ShapeHmmAdapter{}
	r.Estimator = obj.Estimator.CloneMatrixBatchEstimator()
	r.updateEstimate()
	return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmmAdapter) ScalarType() ScalarType {
	return obj.Estimator.ScalarType()
}

func (obj *ShapeHmmAdapter) Dims() (int, int) {
	return obj.Estimator.Dims()
}

func (obj *ShapeHmmAdapter) GetParameters() Vector {
	return obj.Estimator.GetParameters()
}

func (obj *ShapeHmmAdapter) SetParameters(parameters Vector) error {
	return obj.Estimator.SetParameters(parameters)
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmmAdapter) newObservation(x ConstMatrix, gamma ConstVector, p ThreadPool) error {
	n, m := x.Dims()
	for k := n / 2; k < n-n/2; k++ {
		i := k - n/2
		j := k - n/2 + n
		if err := obj.Estimator.NewObservation(x.ConstSlice(i, j, 0, m), gamma.ConstAt(k), p); err != nil {
			return err
		}
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmmAdapter) SetData(x []ConstMatrix, n int) error {
	for d := 0; d < len(x); d++ {
		_, m1 := x[d].Dims()
		_, m2 := obj.Dims()
		if m1 != m2 {
			return fmt.Errorf("data has invalid dimension, expected dimension `%d' but data has dimension `%d'", m2, m1)
		}
	}
	obj.x = x
	obj.n = n
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ShapeHmmAdapter) updateEstimate() error {
	if d, err := obj.Estimator.GetEstimate(); err != nil {
		return err
	} else {
		obj.estimate = d
	}
	return nil
}

func (obj *ShapeHmmAdapter) Estimate(gamma ConstVector, p ThreadPool) error {
	x := obj.x
	for d := 0; d < len(x); d++ {
		// number of observations
		n, _ := x[d].Dims()
		if err := obj.newObservation(x[d], gamma.ConstSlice(0, n), p); err != nil {
			return err
		}
		gamma = gamma.ConstSlice(n, gamma.Dim())
	}
	if err := obj.updateEstimate(); err != nil {
		return err
	}
	return nil
}

func (obj *ShapeHmmAdapter) EstimateOnData(x []ConstMatrix, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *ShapeHmmAdapter) GetEstimate() (MatrixPdf, error) {
	if obj.estimate == nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.estimate, nil
}
