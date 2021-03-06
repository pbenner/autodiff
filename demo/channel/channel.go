/* Copyright (C) 2015-2020 Philipp Benner
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

//import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/newton"
import   "github.com/pbenner/autodiff/algorithm/blahut"

import   "gonum.org/v1/plot"
import   "gonum.org/v1/plot/plotter"
import   "gonum.org/v1/plot/plotutil"
import   "gonum.org/v1/plot/vg"

/* plot methods
 * -------------------------------------------------------------------------- */

type line struct {
  values []float64
  name   string
}

func plotGradientNorm(args ...line) {
  var list []interface{}

  for _, arg := range args {
    xy := make(plotter.XYs, len(arg.values))
    for i := 0; i < len(arg.values); i++ {
      xy[i].X = float64(i)+1
      xy[i].Y = arg.values[i]
    }
    list = append(list, arg.name)
    list = append(list, xy)
  }

  p, err := plot.New()
  if err != nil {
    panic(err)
  }
  p.Title.Text = ""
  p.X.Label.Text = "iteration"
  p.Y.Label.Text = "distance to optimum"
  p.X.Scale = plot.LogScale{}
  p.Y.Scale = plot.LogScale{}
  p.X.Tick.Marker = plot.LogTicks{}
  p.Y.Tick.Marker = plot.LogTicks{}
  p.Legend.Top = true

  err = plotutil.AddLines(p, list...)
  if err != nil {
    panic(err)
  }

  if err := p.Save(8*vg.Inch, 4*vg.Inch, "channel.png"); err != nil {
    panic(err)
  }

}

/* utility
 * -------------------------------------------------------------------------- */

func flatten(m [][]float64) []float64 {
  v := []float64{}
  for i, _ := range m {
    v = append(v, m[i]...)
  }
  return v
}

func distance(v1, v2 []float64) float64 {
  sum := 0.0
  for i, _ := range v1 {
    sum += math.Pow(v1[i]-v2[i], 2.0)
  }
  return math.Sqrt(sum)
}

/* hooks for keeping track of convergence speed
 * -------------------------------------------------------------------------- */

func hook_f(trace *[]float64, pxstar []float64, gradient ConstMatrix, variables ConstVector, s ConstVector) bool {
  n  := (variables.Dim() - 1)/2
  px := make([]float64, n)
  // convert variables to probabilities
  for i := 0; i < n; i++ {
    px[i] = math.Exp(variables.ConstAt(i).GetFloat64())
  }
  // distance to optimum
  d := distance(px, pxstar)
  // append result to trace
  *trace = append(*trace, d)

  return false
}

func hook_b(trace *[]float64, pxstar []float64, px []float64) bool {
  // distance to optimum
  d := distance(px, pxstar)
  // append result to trace
  *trace = append(*trace, d)

  return false
}

/* objective functions for gradient based maximization
 * -------------------------------------------------------------------------- */

func objective_f(channel ConstMatrix, variables ConstVector) MagicScalar {
  n, m := channel.Dims()
  if variables.Dim() != n+1 {
    panic("Input vector has invalid dimension!")
  }
  lambda := variables.ConstAt(n)
  result := NewReal64(0.0)
  sum := NewReal64(0.0)
  t   := NewReal64(0.0)
  pxy := NewReal64(0.0)
  px  := NullDenseReal64Vector(n)
  py  := NullDenseReal64Vector(m)
  // convert variables to probabilities
  for i := 0; i < n; i++ {
    px.At(i).Exp(variables.ConstAt(i))
  }
  // compute p(y) from p(y|x)*p(x)
  for j := 0; j < m; j++ {
    py.At(j).SetFloat64(0.0)
    for i := 0; i < n; i++ {
      py.At(j).Add(py.ConstAt(j), t.Mul(channel.ConstAt(i, j), px.ConstAt(i)))
    }
  }
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      // compute joint probability
      pxy.Mul(channel.ConstAt(i, j), px.ConstAt(i))
      // check if p(x) is zero
      if px.At(i).GetFloat64() == 0.0 {
        result.Add(result, pxy)
      } else {
        // compute p(x,y) log p(x,y)/(p(x)p(y))
        t.Mul(px.ConstAt(i), py.ConstAt(j))
        result.Add(result,
          t.Mul(pxy, t.Log(t.Div(pxy, t))))
      }
    }
  }
  // take logarithm for numerical reasons
  result.Log(result)
  // negate objective function to compute its maximum
  result.Neg(result)
  // add constraint
  for i := 0; i < n; i++ {
    sum.Add(sum, px.At(i))
  }
  result.Add(result, t.Mul(lambda, t.Sub(sum, ConstFloat64(1.0))))

  return result
}

/* main function
 * -------------------------------------------------------------------------- */

func channel_capacity(channel [][]float64, pxstar, px0 []float64) ([][]float64) {
  n := len(channel)
  m := len(channel[0])
  // precision
  const epsilon = 1e-12
  // initial gradient step size
  const step    = 0.001

  // copy variables for automatic differentation
  channelm := NewDenseFloat64Matrix(flatten(channel), n, m)
  // add 1 lagrange multipliers
  px0m     := NewDenseFloat64Vector(append(px0, 2.5))

  // keep track of the path of an algorithm
  trace := make([][]float64, 3)
  trace[0] = []float64{distance(px0, pxstar)}
  trace[1] = []float64{distance(px0, pxstar)}
  trace[2] = []float64{distance(px0, pxstar)}

  // hooks
  hook1 := func(variables ConstVector, gradient ConstMatrix, s ConstVector) bool {
    return hook_f(&trace[0], pxstar, gradient, variables, s)
  }
  hook2 := func(px []float64, J float64) bool {
    return hook_b(&trace[1], pxstar, px)
  }
  hook3 := func(px []float64, J float64) bool {
    return hook_b(&trace[2], pxstar, px)
  }

  // objective function
  f := func(px ConstVector) (MagicScalar, error) { return objective_f(channelm, px), nil }

  // execute algorithms
  _, err := newton.RunCrit(f, px0m,
    newton.HookCrit{hook1},
    newton.Epsilon{epsilon})
  blahut.RunNaive(channel, px0, 500,
    blahut.HookNaive{hook2},
    blahut.Lambda{1.0})
  blahut.RunNaive(channel, px0,  40,
    blahut.HookNaive{hook3},
    blahut.Lambda{8.0})

  if err != nil {
    panic(err)
  }

  return trace
}

func main() {

  channel := [][]float64{
    {0.60, 0.30, 0.10},
    {0.70, 0.10, 0.20},
    {0.50, 0.05, 0.45} }
  pxstar  := []float64{5.017355e-01, 0.0, 4.982645e-01}

  // initial value
  px0 := []float64{1.0/3.0, 1.0/3.0, 1.0/3.0}

  trace := channel_capacity(channel, pxstar, px0)

  plotGradientNorm(
    line{trace[0], "Newton"},
    line{trace[1], "Blahut"},
    line{trace[2], "PP-Blahut"})
}
