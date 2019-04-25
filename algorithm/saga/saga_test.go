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

func hook(x, g ConstVector, y ConstScalar) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("g: %v\n", g)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {

  // x
  x := make([]float64, 2)
  cellSize  := []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
  cellShape := []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
  // y
  class := []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}

  theta_0 := NewVector(RealType, []float64{-1, 0.0, 0.0})
  lr, _   := NewLogisticRegression(theta_0)

  f := func(i int, theta Vector, r Scalar) error {
    if i >= len(cellSize) {
      return fmt.Errorf("index out of bounds")
    }
    if err := lr.SetParameters(theta); err != nil {
      return err
    }
    x[0] = cellSize [i]
    x[1] = cellShape[i]
    if err := lr.ClassLogPdf(r, DenseConstRealVector(x), class[i] == 1); err != nil {
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

  if r, err := Run(Wrapper(f), len(cellSize), theta_0, Hook{hook}, Gamma{1.0/20}, Epsilon{1e-8}, L1Regularization{0.0}, L2Regularization{0.0}); err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
      test.Error("test failed")
    }
  }
}
