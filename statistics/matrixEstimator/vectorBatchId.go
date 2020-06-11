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
import "github.com/pbenner/autodiff/statistics/matrixDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type VectorBatchId struct {
	Estimators []VectorBatchEstimator
	estimate   MatrixPdf
}

/* -------------------------------------------------------------------------- */

func NewVectorBatchId(estimators ...VectorBatchEstimator) (*VectorBatchId, error) {
	e := make([]VectorBatchEstimator, len(estimators))
	for i := 0; i < len(estimators); i++ {
		if estimators[i] == nil {
			return nil, fmt.Errorf("estimator must not be nil")
		}
		e[i] = estimators[i].CloneVectorBatchEstimator()
	}
	r := VectorBatchId{}
	r.Estimators = e
	if err := r.updateEstimate(); err != nil {
		return nil, err
	}
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *VectorBatchId) Clone() *VectorBatchId {
	r := VectorBatchId{}
	r.Estimators = make([]VectorBatchEstimator, len(obj.Estimators))
	for i, estimator := range obj.Estimators {
		r.Estimators[i] = estimator.CloneVectorBatchEstimator()
	}
	r.updateEstimate()
	return &r
}

func (obj *VectorBatchId) CloneMatrixBatchEstimator() MatrixBatchEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *VectorBatchId) ScalarType() ScalarType {
	if len(obj.Estimators) == 0 {
		return nil
	} else {
		return obj.Estimators[0].ScalarType()
	}
}

func (obj *VectorBatchId) Dims() (int, int) {
	if len(obj.Estimators) == 0 {
		return 0, 0
	} else {
		return len(obj.Estimators), obj.Estimators[0].Dim()
	}
}

func (obj *VectorBatchId) GetParameters() Vector {
	if len(obj.Estimators) == 0 {
		return nil
	}
	p := obj.Estimators[0].GetParameters()
	for i := 1; i < len(obj.Estimators); i++ {
		p = p.AppendVector(obj.Estimators[i].GetParameters())
	}
	return p
}

func (obj *VectorBatchId) SetParameters(parameters Vector) error {
	for i := 0; i < len(obj.Estimators); i++ {
		n := obj.Estimators[i].GetParameters().Dim()
		if parameters.Dim() < n {
			return fmt.Errorf("invalid set of parameters")
		}
		if err := obj.Estimators[i].SetParameters(parameters.Slice(0, n)); err != nil {
			return err
		}
		parameters = parameters.Slice(n, parameters.Dim())
	}
	if parameters.Dim() != 0 {
		return fmt.Errorf("invalid set of parameters")
	}
	return nil
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *VectorBatchId) Initialize(p ThreadPool) error {
	for _, estimator := range obj.Estimators {
		if err := estimator.Initialize(p); err != nil {
			return err
		}
	}
	return nil
}

func (obj *VectorBatchId) NewObservation(x ConstMatrix, gamma ConstScalar, p ThreadPool) error {
	n1, m1 := x.Dims()
	n2, m2 := obj.Dims()
	if n1 != n2 || m1 != m2 {
		return fmt.Errorf("data has invalid dimension")
	}
	for i, estimator := range obj.Estimators {
		if err := estimator.NewObservation(x.ConstRow(i), gamma, p); err != nil {
			return err
		}
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *VectorBatchId) updateEstimate() error {
	r := make([]VectorPdf, len(obj.Estimators))
	for i, estimator := range obj.Estimators {
		if d, err := estimator.GetEstimate(); err != nil {
			return err
		} else {
			r[i] = d
		}
	}
	if s, err := matrixDistribution.NewVectorId(r...); err != nil {
		return err
	} else {
		obj.estimate = s
	}
	return nil
}

func (obj *VectorBatchId) GetEstimate() (MatrixPdf, error) {
	if err := obj.updateEstimate(); err != nil {
		return nil, err
	}
	return obj.estimate, nil
}
