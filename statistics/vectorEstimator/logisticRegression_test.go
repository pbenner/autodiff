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
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func hook(x ConstVector, step, y ConstScalar, i int) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("s: %v\n", step)
  fmt.Printf("i: %d\n", i)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func eval_l2_solution(class []float64, x []ConstVector, theta ConstVector, C float64) ConstVector {
  v := AsDenseRealVector(theta)
  v.Variables(1)
  t := NewReal(0.0)
  s := NewReal(0.0)
  l := ConstReal(1.0/(C*float64(len(x))))
  if r, err := vectorDistribution.NewLogisticRegression(v); err != nil {
    panic(err)
  } else {
    for i, _ := range x {
      if err := r.ClassLogPdf(t, x[i].ConstSlice(1,4), class[i] == 1.0); err != nil {
        panic(err)
      }
      s.Add(s, t)
    }
    s.Div(s, ConstReal(1.0/float64(len(x))))
    t.Vnorm(v)
    t.Mul(l, t)
    s.Add(s, t)
  }
  return DenseGradient{s}
}

/* -------------------------------------------------------------------------- */

func TestLogistic1(test *testing.T) {

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

func TestLogistic2(test *testing.T) {

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

func TestLogistic3(test *testing.T) {

  C := 0.1

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
  //estimator.Hook  = hook
  estimator.L2Reg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r := estimator.LogisticRegression.GetParameters()
  z := DenseConstRealVector([]float64{-2.35902836, 0.24435153, 0.26729412})
  t := NullReal()

  fmt.Println(eval_l2_solution(class, x, r, C))
  fmt.Println(eval_l2_solution(class, x, z, C))

  if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}
