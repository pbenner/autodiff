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

import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func hook(x ConstVector, step, y ConstScalar) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("s: %v\n", step)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {

  // data
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
  // x
  x := make([]ConstVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewDenseBareRealVector([]float64{class[i], 1.0, cellSize[i]-1.0, cellShape[i]-1.0})
  }

  estimator, err := NewLogisticRegression(nil, []float64{-1, 0.0, 0.0}, 3)
  if err != nil {
    test.Error(err); return
  }
  //estimator.Hook = hook

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r := estimator.LogisticRegression.GetParameters()
  z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
  t := NullReal()

  if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func Test2(test *testing.T) {

  // data
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
  // x
  x := make([]ConstVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewSparseBareRealVector([]int{0, 1, 2, 3}, []float64{class[i], 1.0, cellSize[i]-1.0, cellShape[i]-1.0}, 4)
  }

  estimator, err := NewLogisticRegression([]int{0, 1, 2}, []float64{-1, 0.0, 0.0}, 3)
  if err != nil {
    test.Error(err); return
  }
  //estimator.Hook = hook

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r := estimator.LogisticRegression.GetParameters()
  z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
  t := NullReal()

  if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}
