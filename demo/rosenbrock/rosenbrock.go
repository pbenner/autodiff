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

package main

/* -------------------------------------------------------------------------- */

import "fmt"
import "os"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/rprop"
import "github.com/pbenner/autodiff/algorithm/bfgs"
import "github.com/pbenner/autodiff/algorithm/newton"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func main() {

	fp1, err := os.Create("rosenbrock.rprop.table")
	if err != nil {
		panic(err)
	}
	defer fp1.Close()

	fp2, err := os.Create("rosenbrock.bfgs.table")
	if err != nil {
		panic(err)
	}
	defer fp2.Close()

	fp3, err := os.Create("rosenbrock.newton.table")
	if err != nil {
		panic(err)
	}
	defer fp3.Close()

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
	hook_rprop := func(gradient, step []float64, x ConstVector, y Scalar) bool {
		fmt.Fprintf(fp1, "%s\n", x.Table())
		fmt.Println("x       :", x)
		fmt.Println("gradient:", gradient)
		fmt.Println("y       :", y)
		fmt.Println()
		return false
	}
	hook_bfgs := func(x, gradient Vector, y Scalar) bool {
		fmt.Fprintf(fp2, "%s\n", x.Table())
		fmt.Println("x       :", x)
		fmt.Println("gradient:", gradient)
		fmt.Println("y       :", y)
		fmt.Println()
		return false
	}
	hook_newton := func(x, gradient Vector, hessian Matrix, y Scalar) bool {
		fmt.Fprintf(fp3, "%s\n", x.Table())
		fmt.Println("x       :", x)
		fmt.Println("gradient:", gradient)
		fmt.Println("y       :", y)
		fmt.Println()
		return false
	}

	x0 := NewVector(RealType, []float64{-0.5, 2})

	rprop.Run(f, x0, 0.05, []float64{1.2, 0.8},
		rprop.Hook{hook_rprop},
		rprop.Epsilon{1e-10})

	bfgs.Run(f, x0,
		bfgs.Hook{hook_bfgs},
		bfgs.Epsilon{1e-10})

	newton.RunMin(f, x0,
		newton.HookMin{hook_newton},
		newton.Epsilon{1e-8},
		newton.HessianModification{"LDL"})

}
