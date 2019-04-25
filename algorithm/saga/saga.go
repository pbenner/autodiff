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

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Gamma struct {
  Value float64
}

type L1Regularization struct {
  Value float64
}

type L2Regularization struct {
  Value float64
}

type Hook struct {
  Value func(ConstVector, ConstVector, ConstScalar) bool
}

type MaxIterations struct {
  Value int
}

type MaxEpochs struct {
  Value int
}

/* -------------------------------------------------------------------------- */

func WrapperDense(f func(int, Vector, Scalar) error) ObjectiveDense {
  y := NullReal()
  w := ConstReal(1.0)
  g := DenseBareRealVector{}
  f_ := func(i int, x Vector) (ConstReal, DenseBareRealVector, ConstReal, error) {
    x.Variables(1)
    if err := f(i, x, y); err != nil {
      return ConstReal(0.0), nil, ConstReal(0.0), err
    }
    if g.Dim() == 0 {
      g = NullDenseBareRealVector(x.Dim())
    }
    g.Set(DenseGradient{y})
    return ConstReal(y.GetValue()), g, w, nil
  }
  return f_
}

/* -------------------------------------------------------------------------- */

func Run(f interface{}, n int, x Vector, args ...interface{}) (Vector, error) {

  hook          := Hook               {   nil}
  epsilon       := Epsilon            {  1e-8}
  gamma         := Gamma              {1.0/30.0}
  maxEpochs     := MaxEpochs          {int(^uint(0) >> 1)}
  maxIterations := MaxIterations      {int(^uint(0) >> 1)}
  l1reg         := L1Regularization   { 0.0}
  l2reg         := L2Regularization   { 0.0}
  inSituDense   := &InSituDense       {}
  inSituSparse  := &InSituSparse      {}

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
    case L1Regularization:
      l1reg = a
    case L2Regularization:
      l2reg = a
    case *InSituDense:
      inSituDense = a
    case *InSituSparse:
      inSituSparse = a
    case InSituDense:
      panic("InSitu must be passed by reference")
    case InSituSparse:
      panic("InSitu must be passed by reference")
    default:
      panic("invalid optional argument")
    }
  }
  if l1reg.Value != 0.0 && l2reg.Value != 0.0 {
    return x, fmt.Errorf("using l1- and l2-regularizations is not supported")
  }
  if l1reg.Value < 0.0 {
    return x, fmt.Errorf("invalid l1-regularization constant")
  }
  if l2reg.Value < 0.0 {
    return x, fmt.Errorf("invalid l2-regularization constant")
  }
  switch g := f.(type) {
  case ObjectiveDense:
    return sagaDense (g, n, x, gamma, epsilon, maxEpochs, maxIterations, l1reg, l2reg, hook, inSituDense)
  case ObjectiveSparse:
    return sagaSparse(g, n, x, gamma, epsilon, maxEpochs, maxIterations, l1reg, l2reg, hook, inSituSparse)
  default:
    panic("invalid objective")
  }
}
