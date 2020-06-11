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

type TranslationEstimator struct {
	ScalarBatchEstimator
	StdEstimator
	c float64
	y Vector
}

/* -------------------------------------------------------------------------- */

func NewTranslationEstimator(estimator ScalarBatchEstimator, pseudocount float64) (*TranslationEstimator, error) {
	r := TranslationEstimator{}
	r.ScalarBatchEstimator = estimator
	r.c = pseudocount
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *TranslationEstimator) Clone() *TranslationEstimator {
	r := TranslationEstimator{}
	r.ScalarBatchEstimator = obj.ScalarBatchEstimator.CloneScalarBatchEstimator()
	r.c = obj.c
	return &r
}

func (obj *TranslationEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

func (obj *TranslationEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *TranslationEstimator) Initialize(p ThreadPool) error {
	obj.y = NullVector(BareRealType, p.NumberOfThreads())
	return obj.ScalarBatchEstimator.Initialize(p)
}

func (obj *TranslationEstimator) NewObservation(x, gamma ConstScalar, p ThreadPool) error {
	y := obj.y.At(p.GetThreadId())
	y.Add(x, ConstReal(obj.c))
	return obj.ScalarBatchEstimator.NewObservation(y, gamma, p)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *TranslationEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
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
	obj.ScalarBatchEstimator.GetEstimate()
	return nil
}

func (obj *TranslationEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *TranslationEstimator) GetEstimate() (ScalarPdf, error) {
	if estimate, err := obj.ScalarBatchEstimator.GetEstimate(); err != nil {
		return nil, err
	} else {
		return scalarDistribution.NewPdfTranslation(estimate, obj.c)
	}
}
