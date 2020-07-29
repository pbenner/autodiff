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

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math/rand"
import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"

/* -------------------------------------------------------------------------- */

func sumOfSquares(x, y Vector, l *Line) MagicScalar {

  s := NewReal64(0.0)
  t := NewReal64(0.0)
  n := NewReal64(float64(x.Dim()))

  for i := 0; i < x.Dim(); i++ {
    s.Add(s, t.Pow(t.Sub(l.Eval(x.At(i)), y.At(i)), ConstFloat64(2)))
  }
  s.Div(s, n)
  return s
}

func gradientDescent(x, y Vector, l *Line) *Line {

  // precision
  const epsilon = 0.00001
  // gradient step size
  const step    = 0.1

  // get a vector of variables
  variables := NullDenseReal64Vector(2)
  variables.At(0).Set(l.Slope())
  variables.At(1).Set(l.Intercept())

  // create the objective function
  f := func(v ConstVector) (MagicScalar, error) {
    l.SetSlope    (v.ConstAt(0))
    l.SetIntercept(v.ConstAt(1))
    return sumOfSquares(x, y, l), nil
  }
//  GradientDescent(f, variables, step, epsilon)
  _, err := rprop.Run(f, variables, step, []float64{1.2, 0.8},
    rprop.Epsilon{epsilon})
  if err != nil {
    panic(err)
  }
  return l
}

func main() {

  const n = 1000
  x := NullDenseReal64Vector(n)
  y := NullDenseReal64Vector(n)

  // random number generator
  r := rand.New(rand.NewSource(42))

  for i := 0; i < n; i++ {
    x.At(i).SetFloat64(r.NormFloat64() + 0)
    y.At(i).SetFloat64(r.NormFloat64() + 2*x.At(i).GetFloat64()+1)
  }

  l := NewLine(NewReal64(-1.23), NewReal64(1.0));
  l  = gradientDescent(x, y, l)

  fmt.Println("slope: ", l.Slope().GetFloat64(), "intercept: ", l.Intercept().GetFloat64())
}
