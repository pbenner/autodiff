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

type StepSize struct {
  Value float64
}

type Beta1 struct {
  Value float64
}

type Beta2 struct {
  Value float64
}

type Epsilon struct {
  Value float64
}

type MaxIterations struct {
  Value int
}

type Hook struct {
  Value func([]float64, ConstVector, ConstScalar) bool
}

type Constraints struct {
  Value func(x Vector) bool
}

type ConstConstraints struct {
  Value func(x ConstVector) bool
}

/* -------------------------------------------------------------------------- */

/* Adam:
 * Kingma, Diederik P., and Jimmy Ba. Adam: A method for stochastic
 * optimization. arXiv preprint arXiv:1412.6980 (2014).
 */

func adam(f func(ConstVector) (MagicScalar, error), x0 ConstVector, step_size, beta1, beta2, epsilon float64,
  maxIterations MaxIterations,
  hook Hook,
  constraints Constraints) (Vector, error) {

  n := x0.Dim()
  // copy variables
  x1 := AsDenseReal64Vector(x0)
  x2 := AsDenseReal64Vector(x0)
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
    // evaluate objective function
    s, err := f(x2)
    if err != nil {
      return x1, err
    }
    // get gradient
    for i := 0; i < n; i++ {
      gradient[i] = s.GetDerivative(i)
    }
    if gradient_is_nan(gradient) {
      return x1, fmt.Errorf("Gradient NaN value detected")
    }
    if (constraints.Value != nil && !constraints.Value(x2)) {
      return x1, fmt.Errorf("Constraints voilated")
    }
    // execute hook if available
    if hook.Value != nil && hook.Value(gradient, x1, s) {
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
      x2.At(i).SetFloat64(x1.Float64At(i) - step_size*m_hat/(math.Sqrt(v_hat) + 1e-8))
      if math.IsNaN(x2.Float64At(i)) {
        return x2, fmt.Errorf("NaN value detected")
      }
      beta1_t *= beta1
      beta2_t *= beta2
    }
    x1.Set(x2)
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

func Run(f interface{}, x0 Vector, step_init float64, eta []float64, args ...interface{}) (Vector, error) {

  hook          := Hook         {nil  }
  step_size     := StepSize     {0.001}
  beta1         := Beta1        {0.9  }
  beta2         := Beta2        {0.999}
  epsilon       := Epsilon      {1e-8 }
  constraints   := Constraints  {nil  }
  maxIterations := MaxIterations{int(^uint(0) >> 1)}

  if len(eta) != 2 {
    panic("Adam(): Argument eta must have length two!")
  }
  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a
    case StepSize:
      step_size = a
    case Beta1:
      beta1 = a
    case Beta2:
      beta2 = a
    case Epsilon:
      epsilon = a
    case Constraints:
      constraints = a
    case MaxIterations:
      maxIterations = a
    default:
      panic("Adam(): Invalid optional argument!")
    }
  }
  switch a := f.(type) {
  case func(ConstVector) (MagicScalar, error):
    return adam(a, x0, step_size.Value, beta1.Value, beta2.Value, epsilon.Value, maxIterations, hook, constraints)
  default:
    panic("invalid objective function")
  }
}

func RunGradient(f interface{}, x0 ConstVector, step_init float64, eta []float64, args ...interface{}) (ConstVector, error) {

  hook          := Hook            {nil  }
  step_size     := StepSize        {0.001}
  beta1         := Beta1           {0.9  }
  beta2         := Beta2           {0.999}
  epsilon       := Epsilon         {1e-8 }
  constraints   := ConstConstraints{nil  }
  maxIterations := MaxIterations   {int(^uint(0) >> 1)}

  if len(eta) != 2 {
    panic("Adam(): Argument eta must have length two!")
  }
  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a
    case Beta1:
      beta1 = a
    case Beta2:
      beta2 = a
    case Epsilon:
      epsilon = a
    case ConstConstraints:
      constraints = a
    case MaxIterations:
      maxIterations = a
    default:
      panic("Adam(): Invalid optional argument!")
    }
  }
  switch a := f.(type) {
  case DenseGradientF:
    return adam_dense_with_gradient(a, x0.(DenseFloat64Vector), step_size.Value, beta1.Value, beta2.Value, epsilon.Value, maxIterations, hook, constraints)
  default:
    panic("invalid objective function")
  }
}
