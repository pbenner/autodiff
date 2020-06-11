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

package scalarEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type DeltaEstimator struct {
	*scalarDistribution.DeltaDistribution
	StdEstimator
}

/* -------------------------------------------------------------------------- */

func NewDeltaEstimator(x float64) (*DeltaEstimator, error) {
	if dist, err := scalarDistribution.NewDeltaDistribution(NewBareReal(x)); err != nil {
		return nil, err
	} else {
		r := DeltaEstimator{}
		r.DeltaDistribution = dist
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *DeltaEstimator) Clone() *DeltaEstimator {
	r := DeltaEstimator{}
	r.DeltaDistribution = obj.DeltaDistribution.Clone()
	return &r
}

func (obj *DeltaEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

func (obj *DeltaEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *DeltaEstimator) Initialize(p ThreadPool) error {
	return nil
}

func (obj *DeltaEstimator) NewObservation(x, gamma ConstScalar, p ThreadPool) error {
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *DeltaEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	return nil
}

func (obj *DeltaEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	return nil
}

func (obj *DeltaEstimator) GetEstimate() (ScalarPdf, error) {
	return obj.DeltaDistribution, nil
}
