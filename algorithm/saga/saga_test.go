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

/* -------------------------------------------------------------------------- */

func hook(x, g Vector, y Scalar) bool {
  fmt.Printf("x: %v\n", x)
  fmt.Printf("g: %v\n", g)
  fmt.Println()
  return false
}

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {

  cellSize  := NewVector(RealType, []float64{
    1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1})
  cellShape := NewVector(RealType, []float64{
    1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1})

  class := NewVector(RealType, []float64{
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0})

  f := func(i int, theta Vector) (Scalar, error) {
    t := NullReal()
    r := NullReal()
    if i >= cellSize.Dim() {
      return nil, fmt.Errorf("index out of bounds")
    }
    // eval linear regression
    r.Set(theta.At(0))
    t.Mul(theta.At(1), cellSize.At(i))
    r.Add(r, t)
    t.Mul(theta.At(2), cellShape.At(i))
    r.Add(r, t)
    // eval logistic model
    r.Neg(r)
    r.Exp(r)
    r.Add(r, ConstReal(1.0))
    r.Div(ConstReal(1.0), r)
    // eval log
    if class.At(i).GetValue() == 0 {
      r.Neg(r)
      r.Log1p(r)
    } else {
      r.Log(r)
    }
    // minimize negative log likelihood
    r.Neg(r)
    if math.IsNaN(r.GetValue()) {
      panic("nan")
    }
    return r, nil
  }
  theta_0 := NewVector(RealType, []float64{-3.549, 0.1841, 0.5067})

  Run(f, cellSize.Dim(), theta_0, Hook{hook})
}
