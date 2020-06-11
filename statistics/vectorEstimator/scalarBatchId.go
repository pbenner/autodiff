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

package vectorEstimator

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ScalarBatchId struct {
	Estimators []ScalarBatchEstimator
	estimate   VectorPdf
}

/* -------------------------------------------------------------------------- */

func NewScalarBatchId(estimators ...ScalarBatchEstimator) (*ScalarBatchId, error) {
	e := make([]ScalarBatchEstimator, len(estimators))
	for i := 0; i < len(estimators); i++ {
		if estimators[i] == nil {
			return nil, fmt.Errorf("estimator must not be nil")
		}
		e[i] = estimators[i].CloneScalarBatchEstimator()
	}
	r := ScalarBatchId{}
	r.Estimators = e
	if err := r.updateEstimate(); err != nil {
		return nil, err
	}
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarBatchId) Clone() *ScalarBatchId {
	r := ScalarBatchId{}
	r.Estimators = make([]ScalarBatchEstimator, len(obj.Estimators))
	for i, estimator := range obj.Estimators {
		r.Estimators[i] = estimator.CloneScalarBatchEstimator()
	}
	r.updateEstimate()
	return &r
}

func (obj *ScalarBatchId) CloneVectorBatchEstimator() VectorBatchEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarBatchId) ScalarType() ScalarType {
	if len(obj.Estimators) == 0 {
		return nil
	} else {
		return obj.Estimators[0].ScalarType()
	}
}

func (obj *ScalarBatchId) Dim() int {
	return len(obj.Estimators)
}

func (obj *ScalarBatchId) GetParameters() Vector {
	if len(obj.Estimators) == 0 {
		return nil
	}
	p := obj.Estimators[0].GetParameters()
	for i := 1; i < len(obj.Estimators); i++ {
		p = p.AppendVector(obj.Estimators[i].GetParameters())
	}
	return p
}

func (obj *ScalarBatchId) SetParameters(parameters Vector) error {
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

func (obj *ScalarBatchId) Initialize(p ThreadPool) error {
	for _, estimator := range obj.Estimators {
		if err := estimator.Initialize(p); err != nil {
			return err
		}
	}
	return nil
}

func (obj *ScalarBatchId) NewObservation(x ConstVector, gamma ConstScalar, p ThreadPool) error {
	if x.Dim() != len(obj.Estimators) {
		return fmt.Errorf("data has invalid dimension")
	}
	for i, estimator := range obj.Estimators {
		if err := estimator.NewObservation(x.ConstAt(i), gamma, p); err != nil {
			return err
		}
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ScalarBatchId) updateEstimate() error {
	r := make([]ScalarPdf, len(obj.Estimators))
	for i, estimator := range obj.Estimators {
		if d, err := estimator.GetEstimate(); err != nil {
			return err
		} else {
			r[i] = d
		}
	}
	if s, err := vectorDistribution.NewScalarId(r...); err != nil {
		return err
	} else {
		obj.estimate = s
	}
	return nil
}

func (obj *ScalarBatchId) GetEstimate() (VectorPdf, error) {
	if err := obj.updateEstimate(); err != nil {
		return nil, err
	}
	return obj.estimate, nil
}
