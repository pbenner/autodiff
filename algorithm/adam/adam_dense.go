/* Copyright (C) 2021 Philipp Benner
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

package adam

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm"

/* -------------------------------------------------------------------------- */

type DenseGradientF func(x, gradient DenseFloat64Vector) error

/* -------------------------------------------------------------------------- */

func adam_dense_with_gradient(evalGradient DenseGradientF, x0 DenseFloat64Vector, step_size, beta1, beta2, epsilon float64,
  maxIterations MaxIterations,
  hook Hook,
  constraints ConstConstraints) (DenseFloat64Vector, error) {

  n := x0.Dim()
  // copy variables
  x1 := x0.Clone()
  x2 := x0.Clone()
  // beta1/2_t variables
  beta1_t := beta1
  beta2_t := beta2
  // step size for each variable
  moment_m := make([]float64, n)
  moment_v := make([]float64, n)
  // gradient
  gradient := NullDenseFloat64Vector(n)
  gradient_is_nan := func(gradient DenseFloat64Vector) bool {
    for i := 0; i < gradient.Dim(); i++ {
      if math.IsNaN(gradient.ConstAt(i).GetFloat64()) {
        return true
      }
    }
    return false
  }
  // check initial value
  if constraints.Value != nil && !constraints.Value(x1) {
    return x1, fmt.Errorf("invalid initial value: %v", x1)
  }
  for i_ := 0; i_ < maxIterations.Value; i_++ {
    // compute partial derivatives and update x
    if err := evalGradient(x2, gradient); err != nil {
      return x1, err
    }
    if gradient_is_nan(gradient) {
      return x1, fmt.Errorf("Gradient NaN value detected")
    }
    if (constraints.Value != nil && !constraints.Value(x2)) {
      return x1, fmt.Errorf("Constraints voilated")
    }
    // execute hook if available
    if hook.Value != nil && hook.Value(gradient, x1, nil) {
      break
    }
    // evaluate stop criterion
    if (Norm(gradient) < epsilon) {
      break
    }
    // update x
    for i := 0; i < n; i++ {
      moment_m[i] = beta1*moment_m[i] + (1.0-beta1)*gradient[i]
      moment_v[i] = beta2*moment_m[i] + (1.0-beta2)*gradient[i]*gradient[i]
      m_hat := moment_m[i]/(1.0 - beta1_t)
      v_hat := moment_v[i]/(1.0 - beta2_t)
      x2[i] = x1[i] - step_size*m_hat/(math.Sqrt(v_hat) + 1e-8)
      if math.IsNaN(x2[i]) {
        return x2, fmt.Errorf("NaN value detected")
      }
      beta1_t *= beta1
      beta2_t *= beta2
    }
    copy(x1, x2)
  }
  return x1, nil
}
