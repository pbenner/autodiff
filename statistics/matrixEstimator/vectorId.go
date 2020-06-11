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

type VectorId struct {
	Estimators []VectorEstimator
	estimate   MatrixPdf
}

/* -------------------------------------------------------------------------- */

func NewVectorId(estimators ...VectorEstimator) (*VectorId, error) {
	e := make([]VectorEstimator, len(estimators))
	for i := 0; i < len(estimators); i++ {
		if estimators[i] == nil {
			return nil, fmt.Errorf("estimator must not be nil")
		}
		e[i] = estimators[i].CloneVectorEstimator()
	}
	r := VectorId{}
	r.Estimators = e
	if err := r.updateEstimate(); err != nil {
		return nil, err
	}
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *VectorId) Clone() *VectorId {
	r := VectorId{}
	r.Estimators = make([]VectorEstimator, len(obj.Estimators))
	for i, estimator := range obj.Estimators {
		r.Estimators[i] = estimator.CloneVectorEstimator()
	}
	r.updateEstimate()
	return &r
}

func (obj *VectorId) CloneMatrixEstimator() MatrixEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *VectorId) ScalarType() ScalarType {
	if len(obj.Estimators) == 0 {
		return nil
	} else {
		return obj.Estimators[0].ScalarType()
	}
}

func (obj *VectorId) Dims() (int, int) {
	if len(obj.Estimators) == 0 {
		return 0, 0
	} else {
		return len(obj.Estimators), obj.Estimators[0].Dim()
	}
}

func (obj *VectorId) GetParameters() Vector {
	if len(obj.Estimators) == 0 {
		return nil
	}
	p := obj.Estimators[0].GetParameters()
	for i := 1; i < len(obj.Estimators); i++ {
		p = p.AppendVector(obj.Estimators[i].GetParameters())
	}
	return p
}

func (obj *VectorId) SetParameters(parameters Vector) error {
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

/* -------------------------------------------------------------------------- */

func (obj *VectorId) SetData(x []ConstMatrix, n int) error {
	if x == nil {
		for _, estimator := range obj.Estimators {
			if err := estimator.SetData(nil, n); err != nil {
				return err
			}
		}
	} else {
		// check data
		for j := 0; j < len(x); j++ {
			n1, m1 := x[j].Dims()
			n2, m2 := obj.Dims()
			if n1 != n2 || m1 != m2 {
				return fmt.Errorf("data has invalid dimension (expected dimension `%dx%d' but data has dimension `%dx%d')", n2, m2, n1, m1)
			}
		}
		for i, estimator := range obj.Estimators {
			// get column i
			y := []ConstVector{}
			for j := 0; j < len(x); j++ {
				y = append(y, x[j].ConstRow(i))
			}
			if err := estimator.SetData(y, n); err != nil {
				return err
			}
		}
	}
	if err := obj.updateEstimate(); err != nil {
		return err
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *VectorId) updateEstimate() error {
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

func (obj *VectorId) Estimate(gamma ConstVector, p ThreadPool) error {
	for _, estimator := range obj.Estimators {
		if err := estimator.Estimate(gamma, p); err != nil {
			return err
		}
	}
	if err := obj.updateEstimate(); err != nil {
		return err
	}
	return nil
}

func (obj *VectorId) EstimateOnData(x []ConstMatrix, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *VectorId) GetEstimate() (MatrixPdf, error) {
	return obj.estimate, nil
}
