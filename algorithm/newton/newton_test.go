/* Copyright (C) 2015, 2017 Philipp Benner
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

package newton

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestNewtonRoot(t *testing.T) {

	f := func(x Vector) (Vector, error) {
		y := NullVector(RealType, 2)
		// y1 = x1^2 + x2^2 - 6
		y.At(0).Sub(Add(Pow(x.At(0), NewReal(2)), Pow(x.At(1), NewReal(2))), NewReal(6))
		// y2 = x1^3 - x2^2
		y.At(1).Sub(Pow(x.At(0), NewReal(3)), Pow(x.At(1), NewReal(2)))

		return y, nil
	}
	v1 := NewVector(RealType, []float64{1, 1})
	v2 := NewVector(RealType, []float64{1.537656, 1.906728})
	v3, err := RunRoot(f, v1, Epsilon{1e-8})
	if err != nil {
		t.Error(err)
	} else {
		if Vnorm(VsubV(v2, v3)).GetValue() > 1e-6 {
			t.Error("Newton method failed!")
		}
	}
}

func TestNewtonCrit1(t *testing.T) {
	f := func(x Vector) (Scalar, error) {
		// minimize x^2 subject to x^2 = 1, which is equivalent to finding
		// the critical points of the lagrangian x^2 + lambda(x^2 - 1)
		y := Add(Mul(x.At(0), x.At(0)), Mul(x.At(1), Sub(Mul(x.At(0), x.At(0)), NewReal(1))))

		return y, nil
	}
	v1 := NewVector(RealType, []float64{3, 5})
	v2 := NewVector(RealType, []float64{1, -1})
	v3, err := RunCrit(f, v1, Epsilon{1e-8})
	if err != nil {
		t.Error(err)
	} else {
		if Vnorm(VsubV(v2, v3)).GetValue() > 1e-6 {
			t.Error("Newton method failed!")
		}
	}
}

func TestNewtonCrit2(t *testing.T) {
	// define Lagrangian function
	f := func(x Vector) (Scalar, error) {
		// minimize x + y subject to x^2 + y^2 = 1, which is equivalent to finding
		// the critical points of the lagrangian x^2 + lambda(x^2 - 1)
		y := Add(Add(x.At(0), x.At(1)), Mul(x.At(2), Sub(Add(Mul(x.At(0), x.At(0)), Mul(x.At(1), x.At(1))), NewReal(1))))

		return y, nil
	}
	v1 := NewVector(RealType, []float64{3, 5, 1})
	v2 := NewVector(RealType, []float64{math.Sqrt(2.0) / 2.0, math.Sqrt(2.0) / 2.0, -math.Sqrt(2.0) / 2.0})
	v3, err := RunCrit(f, v1, Epsilon{1e-8})
	if err != nil {
		t.Error(err)
	} else {
		if Vnorm(VsubV(v2, v3)).GetValue() > 1e-6 {
			t.Error("Newton method failed!")
		}
	}
}
