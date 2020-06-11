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

type GeometricEstimator struct {
	*scalarDistribution.GeometricDistribution
	StdEstimator
	// state
	sum_m []float64
	sum_g []float64
	sum_c []int
}

/* -------------------------------------------------------------------------- */

func NewGeometricEstimator(p float64) (*GeometricEstimator, error) {
	if dist, err := scalarDistribution.NewGeometricDistribution(NewBareReal(p)); err != nil {
		return nil, err
	} else {
		r := GeometricEstimator{}
		r.GeometricDistribution = dist
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *GeometricEstimator) Clone() *GeometricEstimator {
	r := GeometricEstimator{}
	r.GeometricDistribution = obj.GeometricDistribution.Clone()
	r.x = obj.x
	return &r
}

func (obj *GeometricEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

func (obj *GeometricEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *GeometricEstimator) Initialize(p ThreadPool) error {
	obj.sum_m = make([]float64, p.NumberOfThreads())
	obj.sum_g = make([]float64, p.NumberOfThreads())
	obj.sum_c = make([]int, p.NumberOfThreads())
	for i := 0; i < p.NumberOfThreads(); i++ {
		obj.sum_m[i] = math.Inf(-1)
		obj.sum_g[i] = math.Inf(-1)
		obj.sum_c[i] = 0
	}
	return nil
}

func (obj *GeometricEstimator) NewObservation(x, gamma ConstScalar, p ThreadPool) error {
	id := p.GetThreadId()
	if gamma == nil {
		x := math.Log(x.GetValue() + 1.0)
		obj.sum_m[id] = LogAdd(obj.sum_m[id], x)
		obj.sum_c[id]++
	} else {
		x := math.Log(x.GetValue() + 1.0)
		g := gamma.GetValue()
		obj.sum_m[id] = LogAdd(obj.sum_m[id], g+x)
		obj.sum_g[id] = LogAdd(obj.sum_g[id], g)
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *GeometricEstimator) updateEstimate() error {
	// sum up partial results
	sum_m := math.Inf(-1)
	sum_g := math.Inf(-1)
	for i := 0; i < len(obj.sum_m); i++ {
		sum_m = LogAdd(sum_m, obj.sum_m[i])
		sum_g = LogAdd(sum_g, obj.sum_g[i])
		sum_g = LogAdd(sum_g, math.Log(float64(obj.sum_c[i])))
	}
	// compute new parameter
	//////////////////////////////////////////////////////////////////////////////
	p := NewScalar(obj.ScalarType(), math.Exp(sum_g-sum_m))

	//////////////////////////////////////////////////////////////////////////////
	if t, err := scalarDistribution.NewGeometricDistribution(p); err != nil {
		return err
	} else {
		*obj.GeometricDistribution = *t
	}
	obj.sum_m = nil
	obj.sum_g = nil
	obj.sum_c = nil
	return nil
}

func (obj *GeometricEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
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

func (obj *GeometricEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *GeometricEstimator) GetEstimate() (ScalarPdf, error) {
	if obj.sum_m != nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.GeometricDistribution, nil
}
