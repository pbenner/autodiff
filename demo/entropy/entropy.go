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

/* Objective:
 * ----------
 * maximize H(x) = - Sum_i p(x_i) log p(x_i)
 * subject to Sum_i p(x_i) = 1
 * =>
 * Find critical points of L(p) = H(x) + lambda (Sum_i p(x_i) - 1)
 * which is equivalent to
 *  i) Finding the roots of the gradient of L(p), i.e. with Newton's method
 * ii) Minimizing the norm of the gradient of L(p)
 */

package main

/* -------------------------------------------------------------------------- */

import "fmt"
import "errors"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/newton"
import "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/simple"

/* gradient based optimization
 * -------------------------------------------------------------------------- */

func hook_g(gradient, step []float64, px ConstVector, s Scalar) bool {
	fmt.Println("px: ", px)
	return false
}

func hook_f(px Vector, gradient Matrix, s Vector) bool {
	fmt.Println("px: ", px)
	return false
}

/* Gradient of L(p) */
func objective_f(px Vector) (Vector, error) {
	n := px.Dim() - 1
	if px.Dim() != n+1 {
		return nil, errors.New("Input vector has invalid dimension!")
	}
	gradient := NullVector(RealType, n+1)
	// derivative with respect to px[i]
	for i := 0; i < n; i++ {
		gradient.At(i).Sub(NewReal(-1), Log(px.At(i)))
		gradient.At(i).Sub(gradient.At(i), px.At(n))
	}
	// derivative with respect to lambda
	gradient.At(n).SetValue(-1.0)
	for i := 0; i < n; i++ {
		gradient.At(n).Add(gradient.At(n), px.At(i))
	}
	return gradient, nil
}

/* Norm of the gradient of L(p) */
func objective_g(px Vector) (Scalar, error) {
	x, err := objective_f(px)
	return Pow(Vnorm(x), NewBareReal(2.0)), err
}

func main() {
	// precision
	const epsilon = 1e-8
	// initial gradient step size
	const step = 0.001

	// initial value for px
	px0v := []float64{0.5, 0.2, 0.3}
	// append initial value for lambda
	px0m := NewVector(RealType, append(px0v, 1))

	fmt.Println("Rprop optimization:")
	pxn1, err1 := rprop.Run(objective_g, px0m, step, []float64{1.2, 0.8},
		rprop.Hook{hook_g},
		rprop.Epsilon{epsilon})
	fmt.Println("Newton optimization:")
	pxn2, err2 := newton.RunRoot(objective_f, px0m,
		newton.HookRoot{hook_f},
		newton.Epsilon{epsilon})

	if err1 != nil {
		panic(err1)
	}
	if err2 != nil {
		panic(err2)
	}

	fmt.Println("Rprop  p(x): ", pxn1)
	fmt.Println("Newton p(x): ", pxn2)
}
