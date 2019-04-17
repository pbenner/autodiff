/* Copyright (C) 2019 Philipp Benner
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

package saga

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import   "math/rand"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type objective func(int, Vector) (Scalar, error)

type Epsilon struct {
  Value float64
}

type Gamma struct {
  Value float64
}

type Hook struct {
  Value func(Vector, Vector, Scalar) bool
}

type MaxIterations struct {
  Value int
}

type MaxEpochs struct {
  Value int
}

type InSitu struct {
  T1 Vector
  T2 Scalar
}

/* -------------------------------------------------------------------------- */

// nomenclature:
// f(x) = y
// g: gradient
// H: Hessian
func saga(
  f objective,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxEpochs MaxEpochs,
  maxIterations MaxIterations,
  hook Hook,
  inSitu *InSitu,
  options []interface{}) (Vector, error) {

  x1 := x.CloneVector()
  x2 := x.CloneVector()
  var y1 Scalar
  var y2 Scalar
  var err error

  // length of gradient
  d := x.Dim()
  // gradient
  g1 := NullDenseBareRealVector(d)
  g2 := NullDenseBareRealVector(d)
  // sum of gradients
  s := NullDenseBareRealVector(d)

  // allocate temporary memory
  if inSitu.T1 == nil {
    inSitu.T1 = NullDenseBareRealVector(d)
  }
  if inSitu.T2 == nil {
    inSitu.T2 = NullBareReal()
  }
  // temporary variables
  t1 := inSitu.T1
  t2 := inSitu.T2

  // initialize s
  for i := 0; i < n; i++ {
    if y1, err = f(i, x1); err != nil {
      return nil, err
    } else {
      if err := CopyGradient(g1, y1); err != nil {
        panic(err)
      }
      s.VaddV(s, g1)
    }
  }
  // compute first step without s using the last data point
  {
    t1.VdivS(g1, ConstReal(gamma.Value))
    x2.VsubV(x1, t1)
  }

  for i_ := 0; i_ < maxIterations.Value && i_/n < maxEpochs.Value; i_++ {
    // execute hook if available
    if hook.Value != nil && hook.Value(x1, g1, y1) {
      break
    }
    // evaluate stop criterion
    t2.Vnorm(g1)
    if t2.GetValue() < epsilon.Value {
      break
    }
    if math.IsNaN(t2.GetValue()) {
      return x1, fmt.Errorf("NaN value detected")
    }
    j := rand.Intn(n)

    y1, err = f(j, x1); if err != nil {
      return x1, err
    }
    y2, err = f(j, x2); if err != nil {
      return x1, err
    }
    if err = CopyGradient(g1, y1); err != nil {
      panic(err)
    }
    if err = CopyGradient(g2, y2); err != nil {
      panic(err)
    }
    t1.VdivS(s, ConstReal(float64(n)))
    t1.VaddV(t1, g2)
    t1.VsubV(t1, g1)
    t1.VdivS(t1, ConstReal(gamma.Value))
    x1.VsubV(x1, t1)

    x1, x2 = x2, x1
    y1, y2 = y2, y1
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

func run(f objective, n int, x Vector, args ...interface{}) (Vector, error) {

  hook                := Hook               {   nil}
  epsilon             := Epsilon            {  1e-8}
  gamma               := Gamma              {1.0/2.0}
  maxEpochs           := MaxEpochs          {int(^uint(0) >> 1)}
  maxIterations       := MaxIterations      {int(^uint(0) >> 1)}
  inSitu              := &InSitu            {}
  options             := make([]interface{}, 0)

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a
    case Epsilon:
      epsilon = a
    case Gamma:
      gamma = a
    case MaxEpochs:
      maxEpochs = a
    case MaxIterations:
      maxIterations = a
    case *InSitu:
      inSitu = a
    case InSitu:
      panic("InSitu must be passed by reference")
    default:
      options = append(options, a)
    }
  }

  return saga(f, n, x, gamma, epsilon, maxEpochs, maxIterations, hook, inSitu, options)
}

/* -------------------------------------------------------------------------- */

func Run(f func(int, Vector) (Scalar, error), n int, x Vector, args ...interface{}) (Vector, error) {

  return run(f, n, x, args...)
}
