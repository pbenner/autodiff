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

type objective func(int, ConstVector, Scalar) error

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

  x = x.CloneVector()
  x.Variables(1)
  y := NullReal()

  // length of gradient
  d := x.Dim()
  // gradient
  var g1 DenseGradient
  var g2 DenseGradient

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

  // sum of gradients
  s    := NullDenseBareRealVector(d)
  dict := make([]*Real, n)
  // initialize s and d
  for i := 0; i < n; i++ {
    if err := f(i, x, y); err != nil {
      return nil, err
    } else {
      dict[i] = y.Clone()
      s.VaddV(s, DenseGradient{y})
    }
  }

  for i_ := 0; i_ < maxIterations.Value && i_/n < maxEpochs.Value; i_++ {
    // execute hook if available
    if hook.Value != nil && hook.Value(x, s, y) {
      break
    }
    // evaluate stop criterion
    if t2.Vnorm(s).GetValue() < epsilon.Value {
      break
    }
    if math.IsNaN(t2.GetValue()) {
      return x, fmt.Errorf("NaN value detected")
    }
    j := rand.Intn(n)

    if err := f(j, x, y); err != nil {
      return x, err
    }
    g1 = DenseGradient{dict[j]}
    g2 = DenseGradient{y}

    t1.VdivS(s , ConstReal(float64(n)))
    t1.VaddV(t1, g2)
    t1.VsubV(t1, g1)
    t1.VmulS(t1, ConstReal(gamma.Value))
    x .VsubV(x , t1)

    // update table
    s.VsubV(s, g1)
    s.VaddV(s, g2)

    // update dictionary
    y, dict[j] = dict[j], y
  }
  return x, nil
}

/* -------------------------------------------------------------------------- */

func run(f objective, n int, x Vector, args ...interface{}) (Vector, error) {

  hook                := Hook               {   nil}
  epsilon             := Epsilon            {  1e-8}
  gamma               := Gamma              {1.0/30.0}
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

func Run(f func(int, ConstVector, Scalar) error, n int, x Vector, args ...interface{}) (Vector, error) {

  return run(f, n, x, args...)
}
