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
import   "github.com/pbenner/autodiff/algorithm/rprop"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func hook(x ConstVector, step ConstScalar, i int) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("s: %v\n", step)
  fmt.Printf("i: %d\n", i)
  fmt.Println()
  return false
}

func rprop_hook(gradient []float64, step []float64, x Vector, value Scalar) bool {
  fmt.Printf("y: %v\n", value)
  fmt.Printf("x: %v\n", x)
  fmt.Printf("g: %v\n", gradient)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func eval_l1_solution(x []ConstVector, theta ConstVector, C float64) Scalar {
  v := AsDenseRealVector(theta)
  v.Variables(1)
  t := NewReal(0.0)
  s := NewReal(0.0)
  l := ConstReal(1.0/C)
  if r, err := vectorDistribution.NewLogisticRegression(v); err != nil {
    panic(err)
  } else {
    for i, _ := range x {
      if err := r.ClassLogPdf(t, x[i].ConstSlice(1,4), x[i].ConstAt(0).GetValue() == 1.0); err != nil {
        panic(err)
      }
      s.Add(s, t)
    }
    s.Neg(s)
    for i := 1; i < v.Dim(); i++ {
      t.Abs(v.At(i))
      t.Mul(t, l)
      s.Add(s, t)
    }
  }
  return s
}

func eval_l2_solution(x []ConstVector, theta ConstVector, C float64) Scalar {
  n := theta.Dim()
  v := AsDenseRealVector(theta)
  v.Variables(1)
  t := NewReal(0.0)
  s := NewReal(0.0)
  l := ConstReal(1.0/C)
  if r, err := vectorDistribution.NewLogisticRegression(v); err != nil {
    panic(err)
  } else {
    for i, _ := range x {
      if err := r.ClassLogPdf(t, x[i].ConstSlice(1,4), x[i].ConstAt(0).GetValue() == 1.0); err != nil {
        panic(err)
      }
      s.Add(s, t)
    }
    s.Neg(s)
    t.Vnorm(v[1:n])
    t.Mul(t, t)
    t.Mul(t, l)
    t.Mul(t, ConstReal(0.5))
    s.Add(s, t)
  }
  return s
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
  r := estimator.GetParameters()
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
  r := estimator.GetParameters()
  z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
  t := NullReal()

  if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func TestLogistic3(test *testing.T) {

  C := 1.0

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
  //estimator.Hook  = hook
  estimator.TiReg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r_saga := estimator.GetParameters()

  objective := func(r Vector) (Scalar, error) {
    return eval_l2_solution(x, r, C), nil
  }
  r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12}); if err != nil {
    panic(err)
  }
  r_sklearn := DenseConstRealVector([]float64{-2.77092384, 0.25099466, 0.40914467})

  t := NullReal()
  s := NullDenseBareRealVector(r_saga.Dim())
  if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
  if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func TestLogistic4(test *testing.T) {

  C := 1.0

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
  //estimator.Hook  = hook
  estimator.L1Reg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r_saga := estimator.GetParameters()

  objective := func(r Vector) (Scalar, error) {
    return eval_l1_solution(x, r, C), nil
  }
  r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12}); if err != nil {
    panic(err)
  }
  r_sklearn := DenseConstRealVector([]float64{-2.63837871, 0.16460826, 0.44788412})

  t := NullReal()
  s := NullDenseBareRealVector(r_saga.Dim())
  if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
  if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func TestLogistic5(test *testing.T) {

  C := 2.5

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
  //estimator.Hook  = hook
  estimator.TiReg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r_saga := estimator.GetParameters()

  objective := func(r Vector) (Scalar, error) {
    return eval_l2_solution(x, r, C), nil
  }
  r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12}); if err != nil {
    panic(err)
  }
  r_sklearn := DenseConstRealVector([]float64{-2.81978662, 0.22409962, 0.45325202})

  t := NullReal()
  s := NullDenseBareRealVector(r_saga.Dim())
  if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
  if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func TestLogistic6(test *testing.T) {

  C := 2.5

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
  //estimator.Hook  = hook
  estimator.L1Reg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r_saga := estimator.GetParameters()

  objective := func(r Vector) (Scalar, error) {
    return eval_l1_solution(x, r, C), nil
  }
  r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12}); if err != nil {
    panic(err)
  }
  r_sklearn := DenseConstRealVector([]float64{-2.76467776, 0.17584927, 0.48174453})

  t := NullReal()
  s := NullDenseBareRealVector(r_saga.Dim())
  if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
  if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}

func TestLogistic7(test *testing.T) {

  C := 1.0

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

  estimator, err := NewLogisticRegression([]int{1}, []float64{-1}, 3)
  if err != nil {
    test.Error(err); return
  }
  //estimator.Hook  = hook
  estimator.L1Reg = 1.0/C

  err = estimator.EstimateOnData(x, nil, ThreadPool{})
  if err != nil {
    test.Error(err); return
  }
  // result and target
  r_saga := estimator.GetParameters()

  objective := func(r Vector) (Scalar, error) {
    return eval_l1_solution(x, r, C), nil
  }
  r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12}); if err != nil {
    panic(err)
  }
  r_sklearn := DenseConstRealVector([]float64{-2.63837871, 0.16460826, 0.44788412})

  t := NullReal()
  s := NullDenseBareRealVector(r_saga.Dim())
  if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
  if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
    test.Error("test failed")
  }
}
