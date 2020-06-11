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

type DenseGradientF func(x, gradient DenseConstRealVector) error

/* -------------------------------------------------------------------------- */

func rprop_dense_with_gradient(evalGradient DenseGradientF, x0 DenseConstRealVector, step_init float64, eta []float64,
	epsilon Epsilon,
	maxIterations MaxIterations,
	hook Hook,
	constraints ConstConstraints) (DenseConstRealVector, error) {

	n := x0.Dim()
	// copy variables
	x1 := x0.Clone()
	x2 := x0.Clone()
	// step size for each variable
	step := make([]float64, n)
	// gradients
	gradient_new := NullDenseConstRealVector(n)
	gradient_old := NullDenseConstRealVector(n)
	// initialize values
	for i := 0; i < x1.Dim(); i++ {
		step[i] = step_init
		gradient_new[i] = 1
		gradient_old[i] = 1
	}
	gradient_is_nan := func(gradient DenseConstRealVector) bool {
		for i := 0; i < gradient.Dim(); i++ {
			if math.IsNaN(gradient.ValueAt(i)) {
				return true
			}
		}
		return false
	}
	// check initial value
	if constraints.Value != nil && !constraints.Value(x1) {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	for i := 0; i < maxIterations.Value; i++ {
		for i := 0; i < x1.Dim(); i++ {
			gradient_old[i] = gradient_new[i]
		}
		// execute hook if available
		if hook.Value != nil && hook.Value(gradient_new, step, x1, nil) {
			break
		}
		for {
			// update x
			for i := 0; i < x1.Dim(); i++ {
				if gradient_new[i] != 0.0 {
					if gradient_new[i] > 0.0 {
						x2[i] = x1[i] - step[i]
					} else {
						x2[i] = x1[i] + step[i]
					}
				}
				if math.IsNaN(x2.ValueAt(i)) {
					return x2, fmt.Errorf("NaN value detected")
				}
			}
			// compute partial derivatives and update x
			if err := evalGradient(x2, gradient_new); err != nil {
				return x1, err
			}
			if gradient_is_nan(gradient_new) ||
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
		copy(x1, x2)
	}
	return x1, nil
}
