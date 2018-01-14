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
import   "github.com/pbenner/autodiff/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type LogTransformEstimator struct {
  ScalarBatchEstimator
  StdEstimator
  y Vector
}

/* -------------------------------------------------------------------------- */

func NewLogTransformEstimator(estimator ScalarBatchEstimator) (*LogTransformEstimator, error) {
  r := LogTransformEstimator{}
  r.ScalarBatchEstimator = estimator
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogTransformEstimator) Clone() *LogTransformEstimator {
  r := LogTransformEstimator{}
  r.ScalarBatchEstimator = obj.ScalarBatchEstimator.CloneScalarBatchEstimator()
  return &r
}

func (obj *LogTransformEstimator) CloneScalarEstimator() ScalarEstimator {
  return obj.Clone()
}

func (obj *LogTransformEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
  return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *LogTransformEstimator) Initialize(p ThreadPool) error {
  obj.y = NullVector(BareRealType, p.NumberOfThreads())
  return obj.ScalarBatchEstimator.Initialize(p)
}

func (obj *LogTransformEstimator) NewObservation(x, gamma Scalar, p ThreadPool) error {
  i := p.GetThreadId()
  return obj.ScalarBatchEstimator.NewObservation(obj.y.At(i).Log(x), gamma, p)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *LogTransformEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  g := p.NewJobGroup()
  x := obj.x

  // initialize estimator
  obj.Initialize(p)

  // compute sigma
  //////////////////////////////////////////////////////////////////////////////
  if gamma == nil {
    if err := p.AddRangeJob(0, x.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      obj.NewObservation(x.At(i), nil, p)
      return nil
    }); err != nil {
      return err
    }
  } else {
    if err := p.AddRangeJob(0, x.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      obj.NewObservation(x.At(i), gamma.At(i), p)
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

func (obj *LogTransformEstimator) EstimateOnData(x Vector, gamma DenseBareRealVector, p ThreadPool) error {
  if err := obj.SetData(x, x.Dim()); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *LogTransformEstimator) GetEstimate() ScalarPdf {
  estimate := obj.ScalarBatchEstimator.GetEstimate()
  r, _ := scalarDistribution.NewPdfLogTransform(estimate)
  return r
}
