/* Copyright (C) 2015 Philipp Benner
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

package rprop

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
	Value float64
}

type MaxIterations struct {
	Value int
}

type Hook struct {
	Value func([]float64, []float64, ConstVector, Scalar) bool
}

type Constraints struct {
	Value func(x Vector) bool
}

type ConstConstraints struct {
	Value func(x ConstVector) bool
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func rprop(f func(Vector) (Scalar, error), x0 ConstVector, step_init float64, eta []float64,
	epsilon Epsilon,
	maxIterations MaxIterations,
	hook Hook,
	constraints Constraints) (Vector, error) {

	n := x0.Dim()
	t := x0.ElementType()
	// copy variables
	x1 := AsDenseRealVector(x0)
	x2 := AsDenseRealVector(x0)
	// step size for each variable
	step := make([]float64, n)
	// gradients
	gradient_new := make([]float64, n)
	gradient_old := make([]float64, n)
	// initialize values
	for i := 0; i < x1.Dim(); i++ {
		step[i] = step_init
		gradient_new[i] = 1
		gradient_old[i] = 1
	}
	if err := x1.Variables(1); err != nil {
		return nil, err
	}
	gradient_is_nan := func(s Scalar) bool {
		for i := 0; i < s.GetN(); i++ {
			if math.IsNaN(s.GetDerivative(i)) {
				return true
			}
		}
		return false
	}
	// check initial value
	if constraints.Value != nil && !constraints.Value(x1) {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	// evaluate objective function
	s, err := f(x1)
	if err != nil {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	if gradient_is_nan(s) {
		return x1, fmt.Errorf("gradient is NaN for initial value: %v", x1)
	}
	for i := 0; i < maxIterations.Value; i++ {
		for i := 0; i < x1.Dim(); i++ {
			gradient_old[i] = gradient_new[i]
		}
		// compute partial derivatives and update x
		for i := 0; i < x1.Dim(); i++ {
			// save derivative
			gradient_new[i] = s.GetDerivative(i)
		}
		// execute hook if available
		if hook.Value != nil && hook.Value(gradient_new, step, x1, s) {
			break
		}
		// evaluate stop criterion
		if Norm(gradient_new) < epsilon.Value {
			break
		}
		// update step size
		for i := 0; i < x1.Dim(); i++ {
			if gradient_new[i] != 0.0 {
				if (gradient_old[i] < 0 && gradient_new[i] < 0) ||
					(gradient_old[i] > 0 && gradient_new[i] > 0) {
					step[i] *= eta[0]
				} else {
					step[i] *= eta[1]
				}
			}
		}
		for {
			// update x
			for i := 0; i < x1.Dim(); i++ {
				if gradient_new[i] != 0.0 {
					if gradient_new[i] > 0.0 {
						x2.At(i).Sub(x1.At(i), NewScalar(t, step[i]))
					} else {
						x2.At(i).Add(x1.At(i), NewScalar(t, step[i]))
					}
				}
				if math.IsNaN(x2.At(i).GetValue()) {
					return x2, fmt.Errorf("NaN value detected")
				}
			}
			// evaluate objective function
			s, err = f(x2)
			if err != nil || gradient_is_nan(s) ||
				(constraints.Value != nil && !constraints.Value(x2)) {
				// if the updated is invalid reduce step size
				for i := 0; i < x1.Dim(); i++ {
					if gradient_new[i] != 0.0 {
						step[i] *= eta[1]
					}
				}
			} else {
				// new position is valid, exit loop
				break
			}
		}
		x1.Set(x2)
	}
	return x1, nil
}

/* -------------------------------------------------------------------------- */

func Run(f interface{}, x0 Vector, step_init float64, eta []float64, args ...interface{}) (Vector, error) {

	hook := Hook{nil}
	epsilon := Epsilon{1e-8}
	constraints := Constraints{nil}
	maxIterations := MaxIterations{int(^uint(0) >> 1)}

	if len(eta) != 2 {
		panic("Rprop(): Argument eta must have length two!")
	}
	for _, arg := range args {
		switch a := arg.(type) {
		case Hook:
			hook = a
		case Epsilon:
			epsilon = a
		case Constraints:
			constraints = a
		case MaxIterations:
			maxIterations = a
		default:
			panic("Rprop(): Invalid optional argument!")
		}
	}
	switch a := f.(type) {
	case func(Vector) (Scalar, error):
		return rprop(a, x0, step_init, eta, epsilon, maxIterations, hook, constraints)
	default:
		panic("invalid objective function")
	}
}

func RunGradient(f interface{}, x0 ConstVector, step_init float64, eta []float64, args ...interface{}) (ConstVector, error) {

	hook := Hook{nil}
	epsilon := Epsilon{1e-8}
	constraints := ConstConstraints{nil}
	maxIterations := MaxIterations{int(^uint(0) >> 1)}

	if len(eta) != 2 {
		panic("Rprop(): Argument eta must have length two!")
	}
	for _, arg := range args {
		switch a := arg.(type) {
		case Hook:
			hook = a
		case Epsilon:
			epsilon = a
		case ConstConstraints:
			constraints = a
		case MaxIterations:
			maxIterations = a
		default:
			panic("Rprop(): Invalid optional argument!")
		}
	}
	switch a := f.(type) {
	case DenseGradientF:
		return rprop_dense_with_gradient(a, x0.(DenseConstRealVector), step_init, eta, epsilon, maxIterations, hook, constraints)
	default:
		panic("invalid objective function")
	}
}
