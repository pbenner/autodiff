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
import   "math"
import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/gradientDescent"
import   "github.com/pbenner/autodiff/algorithm/rprop"

import   "github.com/gonum/plot"
import   "github.com/gonum/plot/plotter"
import   "github.com/gonum/plot/plotutil"
import   "github.com/gonum/plot/vg"


/* -------------------------------------------------------------------------- */

func plotGradientNorm(gn1, gn2 []float64) {
  xy1 := make(plotter.XYs, len(gn1))
  xy2 := make(plotter.XYs, len(gn2))

  for i := 0; i < len(gn1); i++ {
    xy1[i].X = float64(i)+1
    xy1[i].Y = gn1[i]
  }
  for i := 0; i < len(gn2); i++ {
    xy2[i].X = float64(i)+1
    xy2[i].Y = gn2[i]
  }

  p, err := plot.New()
  if err != nil {
    panic(err)
  }
  p.Title.Text = "Norm of the gradient"
  p.X.Label.Text = "iteration"
  p.Y.Label.Text = "||·||"
  p.X.Scale = plot.LogScale{}
  p.Y.Scale = plot.LogScale{}
  p.X.Tick.Marker = plot.LogTicks{}
  p.Y.Tick.Marker = plot.LogTicks{}
  p.Legend.Top = true

  err = plotutil.AddLines(p,
    "vanilla", xy1,
    "rprop",   xy2)
  if err != nil {
    panic(err)
  }

  if err := p.Save(8*vg.Inch, 4*vg.Inch, "example1.png"); err != nil {
    panic(err)
  }

}

/* -------------------------------------------------------------------------- */

func norm(v []float64) float64 {
  sum := 0.0
  for _, x := range v {
    sum += math.Pow(x, 2.0)
  }
  return math.Sqrt(sum)
}

func hook(err *[]float64, gradient []float64, px Vector, s Scalar) bool {
  *err = append(*err, norm(gradient))
  return false
}

/* -------------------------------------------------------------------------- */

func main() {
  f := func(x Vector) (Scalar, error) {
    // x^4 - 3x^3 + 2
    return Add(Sub(Pow(x[0], NewBareReal(4)), Mul(NewReal(3), Pow(x[0], NewBareReal(3)))), NewReal(2)), nil
  }
  err1 := make([]float64, 0)
  err2 := make([]float64, 0)
  x0 := NewVector(RealType, []float64{4})
  // vanilla gradient descent
  xn1, _ := gradientDescent.Run(f, x0, 0.0001,
    gradientDescent.Hook{func(gradient []float64, px Vector, s Scalar) bool { return hook(&err1, gradient, px, s) }},
    gradientDescent.Epsilon{1e-8})
  // resilient backpropagation
  xn2, _ := rprop.Run(f, x0, 0.0001, []float64{1.2, 0.8},
    rprop.Hook{func(gradient, step []float64, px Vector, s Scalar) bool { return hook(&err2, gradient, px, s) }},
    rprop.Epsilon{1e-8})

  fmt.Println(xn1)
  fmt.Println(xn2)

  plotGradientNorm(err1, err2)
}
