/* Copyright (C) 2016, 2017 Philipp Benner
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

package bfgs

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "os"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestBfgsMatyas(t *testing.T) {

	fp, err := os.Create("bfgs_test1.table")
	if err != nil {
		panic(err)
	}
	defer fp.Close()

	f := func(x Vector) (Scalar, error) {
		// f(x1, x2) = 0.26(x1^2 + x2^2) - 0.48 x1 x2
		// minimum: f(x1,x2) = f(0, 0) = 0
		y := Sub(Mul(NewReal(0.26), Add(Mul(x.At(0), x.At(0)), Mul(x.At(1), x.At(1)))),
			Mul(NewReal(0.48), Mul(x.At(0), x.At(1))))
		return y, nil
	}
	// hook := func(x, gradient Vector, y Scalar) bool {
	//   fmt.Fprintf(fp, "%s\n", x.Table())
	//   fmt.Println("x       :", x)
	//   fmt.Println("gradient:", gradient)
	//   fmt.Println("y       :", y)
	//   fmt.Println()
	//   return false
	// }

	x0 := NewVector(RealType, []float64{-2.5, 2})
	xr := NewVector(RealType, []float64{0, 0})
	xn, err := Run(f, x0,
		//Hook{hook},
		Epsilon{1e-8})
	if err != nil {
		t.Error(err)
	}
	if Vnorm(VsubV(xn, xr)).GetValue() > 1e-6 {
		t.Error("BFGS Matyas test failed!")
	}
}

func TestBfgsRosenbrock(t *testing.T) {

	fp, err := os.Create("bfgs_test2.table")
	if err != nil {
		panic(err)
	}
	defer fp.Close()

	f := func(x Vector) (Scalar, error) {
		// f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
		// a = 1
		// b = 100
		// minimum: (x1,x2) = (a, a^2)
		a := NewReal(1.0)
		b := NewReal(100.0)
		s := Pow(Sub(a, x.At(0)), NewReal(2.0))
		t := Mul(b, Pow(Sub(x.At(1), Mul(x.At(0), x.At(0))), NewReal(2.0)))
		return Add(s, t), nil
	}
	// hook := func(x, gradient Vector, y Scalar) bool {
	//   fmt.Fprintf(fp, "%s\n", x.Table())
	//   fmt.Println("x       :", x)
	//   fmt.Println("gradient:", gradient)
	//   fmt.Println("y       :", y)
	//   fmt.Println()
	//   return false
	// }

	x0 := NewVector(RealType, []float64{-0.5, 2})
	xr := NewVector(RealType, []float64{1, 1})
	xn, err := Run(f, x0,
		//Hook{hook},
		Epsilon{1e-10})
	if err != nil {
		t.Error(err)
	}
	if Vnorm(VsubV(xn, xr)).GetValue() > 1e-8 {
		t.Error("BFGS Rosenbrock test failed!")
	}
}
