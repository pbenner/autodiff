/* Copyright (C) 2017-2020 Philipp Benner
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

import   "fmt"
import   "os"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"
import   "github.com/pbenner/autodiff/algorithm/bfgs"
import   "github.com/pbenner/autodiff/algorithm/newton"

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

  f := func(x ConstVector) (MagicScalar, error) {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a := ConstFloat64(  1.0)
    b := ConstFloat64(100.0)
    c := ConstFloat64(  2.0)
    s := NullReal64()
    t := NullReal64()
    s.Pow(s.Sub(a, x.ConstAt(0)), c)
    t.Mul(b, t.Pow(t.Sub(x.ConstAt(1), t.Mul(x.ConstAt(0), x.ConstAt(0))), c))
    s.Add(s, t)
    return s, nil
  }
  hook_rprop := func(gradient, step []float64, x ConstVector, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }
  hook_bfgs := func(x, gradient ConstVector, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }
  hook_newton := func(x, gradient ConstVector, hessian ConstMatrix, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }

  x0 := NewDenseFloat64Vector([]float64{-0.5, 2})

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
