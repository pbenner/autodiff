/* Copyright (C) 2016-2020 Philipp Benner
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
import   "os"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestBfgsMatyas(test *testing.T) {

  fp, err := os.Create("bfgs_test1.table")
  if err != nil {
    panic(err)
  }
  defer fp.Close()

  f := func(x ConstVector) (MagicScalar, error) {
    // f(x1, x2) = 0.26(x1^2 + x2^2) - 0.48 x1 x2
    // minimum: f(x1,x2) = f(0, 0) = 0
    a  := ConstFloat64(0.26)
    b  := ConstFloat64(0.48)
    t1 := NullReal64()
    t2 := NullReal64()
    t1.Mul(x.ConstAt(0), x.ConstAt(0))
    t2.Mul(x.ConstAt(1), x.ConstAt(1))
    t2.Sub(t1.Mul(a, t1.Add(t1, t2)),
           t2.Mul(b, t2.Mul(x.ConstAt(0), x.ConstAt(1))))
    return t2, nil
  }
  // hook := func(x, gradient ConstVector, y ConstScalar) bool {
  //   fmt.Fprintf(fp,  "%s\n", x.Table())
  //   fmt.Println("x       :", x)
  //   fmt.Println("gradient:", gradient)
  //   fmt.Println("y       :", y)
  //   fmt.Println()
  //   return false
  // }

  t  := NewFloat64(0.0)
  x0 := NewDenseFloat64Vector([]float64{-2.5,2})
  xr := NewDenseFloat64Vector([]float64{0, 0})
  xn, err := Run(f, x0,
    //Hook{hook},
    Epsilon{1e-8})
  if err != nil {
    test.Error(err)
  }
  if t.Vnorm(xn.VsubV(xn, xr)).GetFloat64() > 1e-6 {
    test.Error("BFGS Matyas test failed!")
  }
  os.Remove("bfgs_test1.table")
}

func TestBfgsRosenbrock(test *testing.T) {

  fp, err := os.Create("bfgs_test2.table")
  if err != nil {
    panic(err)
  }
  defer fp.Close()

  f := func(x ConstVector) (MagicScalar, error) {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a  := ConstFloat64(  1.0)
    b  := ConstFloat64(100.0)
    c  := ConstFloat64(  2.0)
    t1 := NullReal64()
    t2 := NullReal64()
    t1.Mul(b, t1.Pow(t1.Sub(x.ConstAt(1), t1.Mul(x.ConstAt(0), x.ConstAt(0))), c))
    t2.Pow(t2.Sub(a, x.ConstAt(0)), c)
    t1.Add(t1, t2)
    return t1, nil
  }
  // hook := func(x, gradient ConstVector, y ConstScalar) bool {
  //   fmt.Fprintf(fp,  "%s\n", x.Table())
  //   fmt.Println("x       :", x)
  //   fmt.Println("gradient:", gradient)
  //   fmt.Println("y       :", y)
  //   fmt.Println()
  //   return false
  // }

  t  := NewFloat64(0.0)
  x0 := NewDenseFloat64Vector([]float64{-0.5, 2})
  xr := NewDenseFloat64Vector([]float64{   1, 1})
  xn, err := Run(f, x0,
    //Hook{hook},
    Epsilon{1e-10})
  if err != nil {
    test.Error(err)
  }
  if t.Vnorm(xn.VsubV(xn, xr)).GetFloat64() > 1e-8 {
    test.Error("BFGS Rosenbrock test failed!")
  }
  os.Remove("bfgs_test2.table")
}
