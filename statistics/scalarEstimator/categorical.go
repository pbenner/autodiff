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
import "math"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type CategoricalEstimator struct {
	*scalarDistribution.CategoricalDistribution
	StdEstimator
	// state
	sum_t [][]float64
	sum_c [][]int
}

/* -------------------------------------------------------------------------- */

func NewCategoricalEstimator(theta []float64) (*CategoricalEstimator, error) {
	t := NewVector(BareRealType, theta)
	if dist, err := scalarDistribution.NewCategoricalDistribution(t); err != nil {
		return nil, err
	} else {
		r := CategoricalEstimator{}
		r.CategoricalDistribution = dist
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *CategoricalEstimator) Clone() *CategoricalEstimator {
	r := CategoricalEstimator{}
	r.CategoricalDistribution = obj.CategoricalDistribution.Clone()
	r.x = obj.x
	return &r
}

func (obj *CategoricalEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

func (obj *CategoricalEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *CategoricalEstimator) Initialize(p ThreadPool) error {
	obj.sum_t = make([][]float64, p.NumberOfThreads())
	obj.sum_c = make([][]int, p.NumberOfThreads())
	for i := 0; i < p.NumberOfThreads(); i++ {
		obj.sum_t[i] = make([]float64, obj.Theta.Dim())
		obj.sum_c[i] = make([]int, obj.Theta.Dim())
		for j := 0; j < obj.Theta.Dim(); j++ {
			obj.sum_t[i][j] = math.Inf(-1)
			obj.sum_c[i][j] = 0
		}
	}
	return nil
}

func (obj *CategoricalEstimator) NewObservation(x, gamma ConstScalar, p ThreadPool) error {
	id := p.GetThreadId()
	i := int(x.GetValue())
	if gamma == nil {
		obj.sum_c[id][i]++
	} else {
		obj.sum_t[id][i] = LogAdd(obj.sum_t[id][i], gamma.GetValue())
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *CategoricalEstimator) updateEstimate() error {
	sum_t := make([]float64, obj.Theta.Dim())
	for j := 0; j < len(sum_t); j++ {
		sum_t[j] = math.Inf(-1)
	}
	// loop over threads
	for i := 0; i < len(obj.sum_t); i++ {
		// loop over categories
		for j := 0; j < len(obj.sum_t[i]); j++ {
			// sum contributions from gamma
			sum_t[j] = LogAdd(sum_t[j], obj.sum_t[i][j])
			// sum contributions from counts
			sum_t[j] = LogAdd(sum_t[j], math.Log(float64(obj.sum_c[i][j])))
		}
	}
	sum := math.Inf(-1)
	// normalize theta
	for j := 0; j < len(sum_t); j++ {
		sum = LogAdd(sum, sum_t[j])
	}
	for j := 0; j < len(sum_t); j++ {
		sum_t[j] = math.Exp(sum_t[j] - sum)
	}
	theta := NewVector(obj.ScalarType(), sum_t)

	if t, err := scalarDistribution.NewCategoricalDistribution(theta); err != nil {
		return err
	} else {
		*obj.CategoricalDistribution = *t
	}
	obj.sum_t = nil
	obj.sum_c = nil
	return nil
}

func (obj *CategoricalEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	g := p.NewJobGroup()
	x := obj.x

	// initialize estimator
	obj.Initialize(p)

	// compute sigma
	//////////////////////////////////////////////////////////////////////////////
	if gamma == nil {
		if err := p.AddRangeJob(0, x.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
			obj.NewObservation(x.ConstAt(i), nil, p)
			return nil
		}); err != nil {
			return err
		}
	} else {
		if err := p.AddRangeJob(0, x.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
			obj.NewObservation(x.ConstAt(i), gamma.ConstAt(i), p)
			return nil
		}); err != nil {
			return err
		}
	}
	if err := p.Wait(g); err != nil {
		return err
	}
	// update estimate
	if err := obj.updateEstimate(); err != nil {
		return err
	}
	return nil
}

func (obj *CategoricalEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *CategoricalEstimator) GetEstimate() (ScalarPdf, error) {
	if obj.sum_t != nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.CategoricalDistribution, nil
}
