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

package saga

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

func hook(x ConstVector, step, y ConstScalar) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("s: %v\n", step)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func Test0(test *testing.T) {
  x := NewDenseRealVector([]float64{1, 1})
  r := NewReal(0.0)
  { // test positive class
    theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
    theta_0.Variables(1)
    lr, _   := NewLogisticRegression(theta_0)

    if err := lr.ClassLogPdf(r, x, true); err != nil {
      test.Error(err); return
    }
    if math.Abs(r.GetValue() - -1.313262e+00) > 1e-4 {
      test.Error("test failed")
    }
    if math.Abs(r.GetDerivative(1) - 0.7310585786300048) > 1e-4 {
      test.Error("test failed")
    }
  }
  { // test negative class
    theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
    theta_0.Variables(1)
    lr, _   := NewLogisticRegression(theta_0)

    if err := lr.ClassLogPdf(r, x, false); err != nil {
      test.Error(err); return
    }
    if math.Abs(r.GetValue() - -0.313262) > 1e-4 {
      test.Error("test failed")
    }
    if math.Abs(r.GetDerivative(1) - -0.268941) > 1e-4 {
      test.Error("test failed")
    }
  }
}

func Test1(test *testing.T) {

  // data
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
  // x
  x := make([]DenseBareRealVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewDenseBareRealVector([]float64{cellSize[i], cellShape[i]})
  }

  theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
  lr, _   := NewLogisticRegression(theta_0)

  f := func(i int, theta Vector, r Scalar) error {
    if i >= len(cellSize) {
      return fmt.Errorf("index out of bounds")
    }
    if err := lr.SetParameters(theta); err != nil {
      return err
    }
    if err := lr.ClassLogPdf(r, x[i], class[i] == 1); err != nil {
      return err
    }
    if math.IsNaN(r.GetValue()) {
      return fmt.Errorf("NaN value detected")
    }
    // minimize negative log likelihood
    r.Neg(r)
    return nil
  }
  z := DenseConstRealVector([]float64{-3.549076e+00, 1.840901e-01, 5.067003e-01})
  t := NullReal()

  if r, err := Run(WrapperDense(f), len(cellSize), theta_0, Hook{}, Gamma{1.0/20}, Epsilon{1e-8}, L1Regularization{0.0}, L2Regularization{0.0}); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
      test.Error("test failed")
    }
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
  x := make([]DenseBareRealVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i], cellShape[i]})
  }

  theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
  lr, _   := NewLogisticRegression(theta_0)
  r       := NewBareReal(0.0)

  f := func(i int, theta, g_old DenseBareRealVector) (ConstReal, DenseBareRealVector, ConstReal, error) {
    y := ConstReal(0.0)
    w := ConstReal(0.0)
    if i >= len(cellSize) {
      return y, nil, w, fmt.Errorf("index out of bounds")
    }
    if err := lr.SetParameters(theta); err != nil {
      return y, nil, w, err
    }
    if err := lr.LogPdf(r, x[i]); err != nil {
      return y, nil, w, err
    }
    if math.IsNaN(r.GetValue()) {
      return y, nil, w, fmt.Errorf("NaN value detected")
    }
    y = ConstReal(r.GetValue())
    w = ConstReal(math.Exp(r.GetValue()) - class[i])
    return y, x[i], w, nil
  }
  z := DenseConstRealVector([]float64{-3.549076e+00, 1.840901e-01, 5.067003e-01})
  t := NullReal()

  if r, err := Run(ObjectiveDense(f), len(cellSize), theta_0, Hook{}, Gamma{1.0/20}, Epsilon{1e-8}, L1Regularization{0.0}, L2Regularization{0.0}); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
      test.Error("test failed")
    }
  }
}

func Test3(test *testing.T) {

  // data
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
  // x
  x := make([]*SparseBareRealVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewSparseBareRealVector([]int{0, 1, 2}, []float64{1.0, cellSize[i], cellShape[i]}, 3)
  }

  theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
  lr, _   := NewLogisticRegression(theta_0)
  r       := NewBareReal(0.0)

  f := func(i int, theta, g_old *SparseBareRealVector) (ConstReal, *SparseBareRealVector, ConstReal, error) {
    y := ConstReal(0.0)
    w := ConstReal(0.0)
    if i >= len(cellSize) {
      return y, nil, w, fmt.Errorf("index out of bounds")
    }
    if err := lr.SetParameters(theta); err != nil {
      return y, nil, w, err
    }
    if err := lr.LogPdf(r, x[i]); err != nil {
      return y, nil, w, err
    }
    if math.IsNaN(r.GetValue()) {
      return y, nil, w, fmt.Errorf("NaN value detected")
    }
    y = ConstReal(r.GetValue())
    w = ConstReal(math.Exp(r.GetValue()) - class[i])
    return y, x[i], w, nil
  }
  z := DenseConstRealVector([]float64{-3.549076e+00, 1.840901e-01, 5.067003e-01})
  t := NullReal()

  if r, err := Run(ObjectiveSparse(f), len(cellSize), theta_0, Hook{}, Gamma{1.0/20}, Epsilon{1e-8}, L1Regularization{0.0}, L2Regularization{0.0}); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
      test.Error("test failed")
    }
  }
}

func Test4(test *testing.T) {

  // data
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
  // x
  x := make([]*SparseBareRealVector, len(cellSize))
  for i := 0; i < len(cellSize); i++ {
    x[i] = NewSparseBareRealVector([]int{0, 1, 2}, []float64{1.0, cellSize[i]-1.0, cellShape[i]-1.0}, 3)
  }

  theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
  lr, _   := NewLogisticRegression(theta_0)
  r       := NewBareReal(0.0)

  f := func(i int, theta, g_old *SparseBareRealVector) (ConstReal, *SparseBareRealVector, ConstReal, error) {
    y := ConstReal(0.0)
    w := ConstReal(0.0)
    if i >= len(cellSize) {
      return y, nil, w, fmt.Errorf("index out of bounds")
    }
    if err := lr.SetParameters(theta); err != nil {
      return y, nil, w, err
    }
    if err := lr.LogPdf(r, x[i]); err != nil {
      return y, nil, w, err
    }
    if math.IsNaN(r.GetValue()) {
      return y, nil, w, fmt.Errorf("NaN value detected")
    }
    y = ConstReal(r.GetValue())
    w = ConstReal(math.Exp(r.GetValue()) - class[i])
    return y, x[i], w, nil
  }
  z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
  t := NullReal()

  if r, err := Run(ObjectiveSparse(f), len(cellSize), theta_0, Hook{}, Gamma{1.0/20}, Epsilon{1e-8}, L1Regularization{0.0}, L2Regularization{0.0}); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
      test.Error("test failed")
    }
  }
}
