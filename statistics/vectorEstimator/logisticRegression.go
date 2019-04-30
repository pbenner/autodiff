/* Copyright (C) 2019 Philipp Benner
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

//import   "fmt"
//import   "math"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type LogisticRegression struct {
  *vectorDistribution.LogisticRegression
  StdEstimator
  Sparse bool
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(index []int, theta_ []float64, n int) (*LogisticRegression, error) {
  r := LogisticRegression{}
  if index == nil {
    r.Sparse = false
    if dist, err := vectorDistribution.NewLogisticRegression(NewDenseBareRealVector(theta_)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  } else {
    r.Sparse = true
    if dist, err := vectorDistribution.NewLogisticRegression(NewSparseBareRealVector(index, theta_, n)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  }
  return nil, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Clone() *LogisticRegression {
  r := LogisticRegression{}
  r.LogisticRegression = obj.LogisticRegression.Clone()
  return &r
}

func (obj *LogisticRegression) CloneVectorEstimator() VectorEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Estimate(gamma ConstVector, p ThreadPool) error {
  return nil
}

func (obj *LogisticRegression) EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error {
  return nil
}

func (obj *LogisticRegression) GetEstimate() (VectorPdf, error) {
  return obj.LogisticRegression, nil
}
