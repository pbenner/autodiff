/* Copyright (C) 2015-2020 Philipp Benner
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

package gradientDescent

/* -------------------------------------------------------------------------- */

import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func([]float64, ConstVector, ConstScalar) bool
}

/* -------------------------------------------------------------------------- */

func gradientDescent(f func(ConstVector) (MagicScalar, error), x0 Vector, step, epsilon float64,
  hook func([]float64, ConstVector, ConstScalar) bool) (Vector, error) {

  // copy variables
  x := AsDenseReal64Vector(x0)
  x.Variables(1)
  // slice containing the gradient
  gradient := make([]float64, x.Dim())

  for {
    // evaluate objective function
    s, err := f(x)
    if err != nil {
      return x, err
    }
    // compute partial derivatives and update variables
    for i := 0; i < x.Dim(); i++ {
      // save partial derivative
      gradient[i] = s.GetDerivative(i)
    }
    // execute hook if available
    if hook != nil && hook(gradient, x, s) {
      break
    }
    // evaluate stop criterion
    if Norm(gradient) < epsilon {
      break
    }
    // update variables
    for i := 0; i < x.Dim(); i++ {
      x.At(i).Sub(x.At(i), ConstFloat64(step*s.GetDerivative(i)))
      if math.IsNaN(x.ConstAt(i).GetFloat64()) {
        panic("Gradient descent diverged!")
      }
    }
  }
  return x, nil
}

/* -------------------------------------------------------------------------- */

func Run(f func(ConstVector) (MagicScalar, error), x0 Vector, step float64, args ...interface{}) (Vector, error) {

  hook    := Hook   { nil}.Value
  epsilon := Epsilon{1e-8}.Value

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a.Value
    case Epsilon:
      epsilon = a.Value
    default:
      panic("GradientDescent(): Invalid optional argument!")
    }
  }
  return gradientDescent(f, x0, step, epsilon, hook)
}
