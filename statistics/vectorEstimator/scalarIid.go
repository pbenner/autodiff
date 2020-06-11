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

type ScalarIid struct {
	Estimator ScalarEstimator
	n         int
}

/* -------------------------------------------------------------------------- */

func NewScalarIid(estimator ScalarEstimator, n int) (*ScalarIid, error) {
	r := ScalarIid{}
	r.Estimator = estimator.CloneScalarEstimator()
	r.n = n
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIid) Clone() *ScalarIid {
	r := ScalarIid{}
	r.Estimator = obj.Estimator.CloneScalarEstimator()
	r.n = obj.n
	return &r
}

func (obj *ScalarIid) CloneVectorEstimator() VectorEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIid) ScalarType() ScalarType {
	return obj.Estimator.ScalarType()
}

func (obj *ScalarIid) Dim() int {
	return obj.n
}

func (obj *ScalarIid) GetParameters() Vector {
	return obj.Estimator.GetParameters()
}

func (obj *ScalarIid) SetParameters(parameters Vector) error {
	return obj.Estimator.SetParameters(parameters)
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIid) SetData(x []ConstVector, n int) error {
	if len(x) == 0 {
		return nil
	}
	m := 0
	for i := 0; i < len(x); i++ {
		m += x[i].Dim()
	}
	if obj.n != -1 && obj.n != m {
		return fmt.Errorf("data has invalid dimension (expected dimension `%d' but data has dimension `%d)", obj.n, m)
	}
	y := NullVector(x[0].ElementType(), m)
	for i, k := 0, 0; i < len(x); i++ {
		for j := 0; j < x[i].Dim(); j++ {
			y.At(k).Set(x[i].ConstAt(j))
			k++
		}
	}
	return obj.Estimator.SetData(y, m)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ScalarIid) Estimate(gamma ConstVector, p ThreadPool) error {
	return obj.Estimator.Estimate(gamma, p)
}

func (obj *ScalarIid) EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *ScalarIid) GetEstimate() (VectorPdf, error) {
	if d, err := obj.Estimator.GetEstimate(); err != nil {
		return nil, err
	} else {
		return vectorDistribution.NewScalarIid(d, obj.n)
	}
}
