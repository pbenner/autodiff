/* Copyright (C) 2017 Philipp Benner
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

/* Reference:
 * Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization 2nd." (2006).
 * Algorithm 3.2
 */

package lineSearch

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type objective func(float64) (float64, float64, error)
type constraints func(float64) bool

type Parameters struct {
	Alpha1  float64
	MaxEval int
}

type Constraints struct {
	Value func(x Scalar) bool
}

type Hook struct {
	Value func(Scalar, Scalar, Scalar) bool
}

func newConstraints(constraints Constraints) constraints {
	a := NewBareReal(0.0)
	r := func(alpha float64) bool {
		if constraints.Value != nil {
			a.SetValue(alpha)
			return constraints.Value(a)
		}
		return true
	}
	return r
}

/* -------------------------------------------------------------------------- */

func quadraticMin(a, fa, fpa, b, fb float64) float64 {
	// f(x) = B*(x-a)^2 + C*(x-a) + D
	D := fa
	C := fpa
	db := b - a*1.0
	B := (fb - D - C*db) / (db * db)
	return a - C/(2.0*B)
}

func zoom(f objective, alpha_lo, alpha_hi, y0, ylo, yhi, g0, glo float64, maxEval int, hook Hook) (float64, error) {

	var alpha_j float64

	// constants for Wolfe conditions
	c1 := 1e-4
	c2 := 0.9

	if maxEval <= 0 {
		return quadraticMin(alpha_lo, ylo, glo, alpha_hi, yhi), nil
	}

	for i := 0; i < maxEval; i++ {

		alpha_j = quadraticMin(alpha_lo, ylo, glo, alpha_hi, yhi)

		if math.IsNaN(alpha_j) || alpha_j < math.Min(alpha_lo, alpha_hi) || alpha_j > math.Max(alpha_lo, alpha_hi) {
			alpha_j = (alpha_lo + alpha_hi) / 2.0
		}

		if alpha_j == 0.0 {
			return alpha_j, fmt.Errorf("line search failed")
		}

		yj, gj, err := f(alpha_j)
		if err != nil {
			return 0.0, err
		}
		// execute hook if available
		if hook.Value != nil && hook.Value(NewBareReal(alpha_j), NewBareReal(yj), NewBareReal(gj)) {
			return alpha_j, nil
		}

		if yj > y0+c1*alpha_j*g0 || yj >= ylo {
			alpha_hi = alpha_j
			yhi = yj
		} else {
			if math.Abs(gj) <= -c2*g0 {
				return alpha_j, nil
			}
			if gj*(alpha_hi-alpha_lo) >= 0.0 {
				alpha_hi = alpha_lo
				yhi = ylo
			}
			alpha_lo = alpha_j
			ylo = yj
			glo = gj
		}
	}
	return alpha_j, nil
}

func lineSearch(f objective,
	parameters Parameters,
	constraints constraints,
	hook Hook) (float64, error) {

	maxEval := parameters.MaxEval

	// constants for Wolfe conditions
	c1 := 1e-4
	c2 := 0.9

	y0, g0, err := f(0.0)
	if err != nil {
		return 0.0, err
	}

	// variables at step i
	yi, gi, alpha_i := y0, g0, 0.0
	// variables at step j = i+1
	yj, gj, alpha_j := 0.0, 0.0, parameters.Alpha1

	for i := 0; i < maxEval; i++ {
		if alpha_j == 0.0 {
			return 0.0, fmt.Errorf("line search failed")
		}
		// decrease alpha_j until constraints are satisfied
		for !constraints(alpha_j) {
			alpha_j *= 0.5
		}
		yj, gj, err = f(alpha_j)
		if err != nil {
			return 0.0, err
		}
		// execute hook if available
		if hook.Value != nil && hook.Value(NewBareReal(alpha_j), NewBareReal(yj), NewBareReal(gj)) {
			return alpha_j, nil
		}

		if yj > y0+c1*alpha_j*g0 || (yj >= yi && i > 0) {
			return zoom(f, alpha_i, alpha_j, y0, yi, yj, g0, gi, maxEval-i, hook)
		}
		if math.Abs(gj) <= -c2*g0 {
			return alpha_j, nil
		}
		if gj >= 0.0 {
			return zoom(f, alpha_j, alpha_i, y0, yj, yi, g0, gj, maxEval-i, hook)
		}
		// select new alpha
		alpha_i = 2.0 * alpha_j
		// swap variables
		yi, yj = yj, yi
		gi, gj = gj, gi
		alpha_i, alpha_j = alpha_j, alpha_i
	}
	return alpha_i, nil
}

/* -------------------------------------------------------------------------- */

func run(f objective, args ...interface{}) (float64, error) {

	parameters := Parameters{1, 20}
	constraints := Constraints{nil}
	hook := Hook{nil}

	for _, arg := range args {
		switch a := arg.(type) {
		case Parameters:
			parameters = a
		case Constraints:
			constraints = a
		case Hook:
			hook = a
		}
	}
	return lineSearch(f, parameters, newConstraints(constraints), hook)
}

/* -------------------------------------------------------------------------- */

func Run(f_ func(Scalar) (Scalar, error), t ScalarType, args ...interface{}) (Scalar, error) {
	// copy of x for computing derivatives
	X := NewReal(0.0)
	// objective function
	f := func(x float64) (float64, float64, error) {
		X.Reset()
		X.SetValue(x)
		if err := Variables(1, X); err != nil {
			return 0, 0, err
		}
		// evaluate objective function
		Y, err := f_(X)
		if err != nil {
			return 0.0, 0.0, err
		} else {
			return Y.GetValue(), Y.GetDerivative(0), nil
		}
	}
	if alpha, err := run(f, args...); err != nil {
		return NewScalar(t, alpha), err
	} else {
		return NewScalar(t, alpha), nil
	}
}
