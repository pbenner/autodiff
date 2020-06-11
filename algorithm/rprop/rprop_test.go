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
import "os"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
	m1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	m2 := m1.CloneMatrix()
	m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

	rows, cols := m1.Dims()
	if rows != cols {
		panic("MInverse(): Not a square matrix!")
	}
	I := IdentityMatrix(m1.ElementType(), rows)
	// objective function
	f := func(x Vector) (Scalar, error) {
		m2.AsVector().Set(x)
		s := Mnorm(MsubM(MdotM(m1, m2), I))
		return s, nil
	}
	x, _ := Run(f, m2.AsVector(), 0.01, []float64{2, 0.1})
	m2.AsVector().Set(x)

	if Mnorm(MsubM(m2, m3)).GetValue() > 1e-8 {
		t.Error("Inverting matrix failed!")
	}
}

/* -------------------------------------------------------------------------- */

func TestRPropRosenbrock(t *testing.T) {

	fp, err := os.Create("rprop_test.table")
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
	hook := func(gradient []float64, step []float64, x ConstVector, value Scalar) bool {
		fmt.Fprintf(fp, "%s\n", x.Table())
		return false
	}

	x0 := NewVector(RealType, []float64{-10, 10})
	xr := NewVector(RealType, []float64{1, 1})
	xn, _ := Run(f, x0, 0.01, []float64{1.2, 0.8},
		Hook{hook},
		Epsilon{1e-10})

	if Vnorm(xr.VsubV(xr, xn)).GetValue() > 1e-8 {
		t.Error("Rosenbrock test failed!")
	}
}

/* -------------------------------------------------------------------------- */

func TestRPropRosenbrockGradient(t *testing.T) {

	f := func(x, gradient DenseConstRealVector) error {
		// f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
		// a = 1
		// b = 100
		// minimum: (x1,x2) = (a, a^2)
		a := 1.0
		b := 100.0
		x1 := x[0]
		x2 := x[1]
		gradient[0] = -2*(a-x1) - 2*b*(x2-x1*x1)*2*x1
		gradient[1] = 2 * b * (x2 - x1*x1)
		return nil
	}
	// hook := func(gradient []float64, step []float64, x ConstVector, value Scalar) bool {
	//   fmt.Printf("%s\n", x.Table())
	//   return false
	// }

	x0 := DenseConstRealVector([]float64{-10, 10})
	xr := NewVector(RealType, []float64{1, 1})
	xn, _ := RunGradient(DenseGradientF(f), x0, 0.01, []float64{1.2, 0.8},
		//Hook{hook},
		Epsilon{1e-10})

	if Vnorm(xr.VsubV(xr, xn)).GetValue() > 1e-8 {
		t.Error("Rosenbrock test failed!")
	}
}
