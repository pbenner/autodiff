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

//import   "fmt"
//import   "math"
import   "testing"

//import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"

import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func TestNormal1(t *testing.T) {

  p := ThreadPool{}

  if estimator, err := NewNormalEstimator([]float64{1,2,3}, []float64{
    1,0,0, 0,1,0, 0,0,1}, 1e-8); err != nil {
    t.Error(err)
  } else {
    estimator.Initialize(p)

    estimator.NewObservation(DenseConstRealVector([]float64{ 1, 3, 2}), nil, p)
    estimator.NewObservation(DenseConstRealVector([]float64{ 2, 4, 1}), nil, p)
    estimator.NewObservation(DenseConstRealVector([]float64{10, 5, 8}), nil, p)
  }
  
}
