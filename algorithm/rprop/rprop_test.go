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

//import   "fmt"
//import   "os"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  m2 := NullDenseReal64Matrix(2, 2)
  m3 := NewDenseFloat64Matrix([]float64{-2, 1, 1.5, -0.5}, 2, 2)
  s  := NewReal64(0.0)

  rows, cols := m1.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := DenseIdentityMatrix(m1.ElementType(), rows)
  // objective function
  f := func(x ConstVector) (MagicScalar, error) {
    m2.AsVector().Set(x)
    m2.MdotM(m1, m2)
    m2.MsubM(m2, I)
    s.Mnorm(m2)
    return s, nil
  }
  x, _ := Run(f, m2.AsVector(), 0.01, []float64{2, 0.1})
  m2.AsVector().Set(x)

  if s.Mnorm(m2.MsubM(m2, m3)).GetFloat64() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

/* -------------------------------------------------------------------------- */

func TestRPropRosenbrock(t *testing.T) {

  // fp, err := os.Create("rprop_test.table")
  // if err != nil {
  //   panic(err)
  // }
  // defer fp.Close()

  f := func(x ConstVector) (MagicScalar, error) {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a := ConstFloat64(  1.0)
    b := ConstFloat64(100.0)
    t := NullReal64()
    s := NullReal64()
    t.Mul(x.ConstAt(0), x.ConstAt(0))
    t.Sub(x.ConstAt(1), t)
    t.Mul(t, t)
    t.Mul(t, b)
    s.Sub(a, x.ConstAt(0))
    s.Mul(s, s)
    s.Add(s, t)
    return s, nil
  }
  // hook := func(gradient []float64, step []float64, x ConstVector, value ConstScalar) bool {
  //   fmt.Fprintf(fp, "%s\n", x.Table())
  //   return false
  // }

  x0 := NewDenseFloat64Vector([]float64{-10,10})
  xr := NewDenseFloat64Vector([]float64{  1, 1})
  s  := NewFloat64(0.0)
  xn, _ := Run(f, x0, 0.01, []float64{1.2, 0.8},
    //Hook{hook},
    Epsilon{1e-10})

  if s.Vnorm(xr.VsubV(xr, xn)).GetFloat64() > 1e-8 {
    t.Error("Rosenbrock test failed!")
  }
}

/* -------------------------------------------------------------------------- */

func TestRPropRosenbrockGradient(t *testing.T) {

  f := func(x, gradient DenseFloat64Vector) error {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a :=   1.0
    b := 100.0
    x1 := x[0]
    x2 := x[1]
    gradient[0] = -2*(a - x1) - 2*b*(x2 - x1*x1)*2*x1
    gradient[1] = 2*b*(x2 - x1*x1)
    return nil
  }
  // hook := func(gradient []float64, step []float64, x ConstVector, value Scalar) bool {
  //   fmt.Printf("%s\n", x.Table())
  //   return false
  // }

  x0 := NewDenseFloat64Vector([]float64{-10,10})
  xr := NewDenseFloat64Vector([]float64{  1, 1})
  s  := NewFloat64(0.0)
  xn, _ := RunGradient(DenseGradientF(f), x0, 0.01, []float64{1.2, 0.8},
    //Hook{hook},
    Epsilon{1e-10})

  if s.Vnorm(xr.VsubV(xr, xn)).GetFloat64() > 1e-8 {
    t.Error("Rosenbrock test failed!")
  }
}
