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

  X := DenseFloat64Matrix{}
  y := DenseFloat64Vector{}
  if err := X.Import("logisticData.X.table"); err != nil {
    panic(err)
  }
  if err := y.Import("logisticData.y.table"); err != nil {
    panic(err)
  }
  n, m := X.Dims()

  z := NullDenseReal64Vector(n)
  r := NullReal64()
  s := NullReal64()
  t := NullReal64()
  f := func(theta ConstVector) (MagicScalar, error) {
    r.SetFloat64(0.0)
    z.MdotV(&X, theta)
    for i := 0; i < n; i++ {
      s.Sigmoid(z[i], t)
      if y[i] == 1.0 {
        s.Log(s)
      } else {
        s.Sub(ConstFloat64(1.0), s)
        s.Log(s)
      }
      r.Add(r, s)
    }
    r.Neg(r)
    return r, nil
  }

  theta0 := NullDenseFloat64Vector(m)

  start := time.Now()
  thetan, _ := Run(f, theta0,
    Epsilon{0.0},
    MaxIterations{1000},
    Epsilon{1e-10})
  elapsed := time.Since(start)

  fmt.Printf("%s\n", thetan)
  fmt.Printf("%s\n", elapsed)
}
