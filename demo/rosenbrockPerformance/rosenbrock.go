/* Copyright (C) 2021 Philipp Benner
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
import   "time"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm/adam"

/* -------------------------------------------------------------------------- */

func main() {

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

  x0 := NewDenseFloat64Vector([]float64{-10,10})

  start := time.Now()
  xn, _ := Run(f, x0,
    Epsilon{0.0},
    MaxIterations{10000},
    Epsilon{1e-10})
  elapsed := time.Since(start)

  fmt.Printf("%v\n", xn)
  fmt.Printf("%s\n", elapsed)
}
