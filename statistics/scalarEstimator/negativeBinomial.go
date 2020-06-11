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

import "fmt"
import "math"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NegativeBinomialEstimator struct {
	*scalarDistribution.NegativeBinomialDistribution
	StdEstimator
	// state
	sum_r []float64
	sum_k []float64
}

/* -------------------------------------------------------------------------- */

func NewNegativeBinomialEstimator(r, p float64) (*NegativeBinomialEstimator, error) {
	if dist, err := scalarDistribution.NewNegativeBinomialDistribution(NewBareReal(r), NewBareReal(p)); err != nil {
		return nil, err
	} else {
		r := NegativeBinomialEstimator{}
		r.NegativeBinomialDistribution = dist
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *NegativeBinomialEstimator) Clone() *NegativeBinomialEstimator {
	r := NegativeBinomialEstimator{}
	r.NegativeBinomialDistribution = obj.NegativeBinomialDistribution.Clone()
	r.x = obj.x
	return &r
}

func (obj *NegativeBinomialEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

func (obj *NegativeBinomialEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *NegativeBinomialEstimator) Initialize(p ThreadPool) error {
	obj.sum_r = make([]float64, p.NumberOfThreads())
	obj.sum_k = make([]float64, p.NumberOfThreads())
	for i := 0; i < p.NumberOfThreads(); i++ {
		obj.sum_r[i] = math.Inf(-1)
		obj.sum_k[i] = math.Inf(-1)
	}
	return nil
}

func (obj *NegativeBinomialEstimator) NewObservation(x, gamma ConstScalar, p ThreadPool) error {
	id := p.GetThreadId()
	if gamma == nil {
		x := x.GetValue()
		r := obj.R.GetValue()
		obj.sum_k[id] = LogAdd(obj.sum_k[id], math.Log(x))
		obj.sum_r[id] = LogAdd(obj.sum_r[id], math.Log(r))
	} else {
		x := x.GetValue()
		g := gamma.GetValue()
		r := obj.R.GetValue()
		obj.sum_k[id] = LogAdd(obj.sum_k[id], g+math.Log(x))
		obj.sum_r[id] = LogAdd(obj.sum_r[id], g+math.Log(r))
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *NegativeBinomialEstimator) updateEstimate() error {
	// sum up partial results
	sum_k := math.Inf(-1)
	sum_r := math.Inf(-1)
	for i := 0; i < len(obj.sum_k); i++ {
		sum_k = LogAdd(sum_k, obj.sum_k[i])
		sum_r = LogAdd(sum_r, obj.sum_r[i])
	}
	if math.IsInf(sum_r, -1) && math.IsInf(sum_k, -1) {
		return fmt.Errorf("negative binomial parameter estimation failed")
	}
	t := obj.ScalarType()
	q := NewScalar(t, 0.0)
	q.SetValue(math.Exp(sum_k - LogAdd(sum_r, sum_k)))
	if t, err := scalarDistribution.NewNegativeBinomialDistribution(obj.R, q); err != nil {
		return err
	} else {
		*obj.NegativeBinomialDistribution = *t
	}
	obj.sum_k = nil
	obj.sum_r = nil
	return nil
}

func (obj *NegativeBinomialEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
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

func (obj *NegativeBinomialEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *NegativeBinomialEstimator) GetEstimate() (ScalarPdf, error) {
	if obj.sum_k != nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.NegativeBinomialDistribution, nil
}
