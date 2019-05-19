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

type TikhonovRegularization struct {
  Value float64
}

type Hook struct {
  Value func(ConstVector, ConstScalar, int) bool
}

type MaxIterations struct {
  Value int
}

type Seed struct {
  Value int64
}

type InSitu struct {
  T1  DenseBareRealVector
  T2 *BareReal
}

/* -------------------------------------------------------------------------- */

func WrapperDense(f func(int, Vector, Scalar) error) Objective1Dense {
  x := NullDenseRealVector(0)
  y := NullReal()
  w := ConstReal(1.0)
  f_ := func(i int, x_ DenseBareRealVector) (ConstReal, ConstReal, DenseConstRealVector, error) {
    if x.Dim() == 0 {
      x = NullDenseRealVector(x_.Dim())
    }
    x.Set(x_)
    x.Variables(1)
    if err := f(i, x, y); err != nil {
      return ConstReal(0.0), ConstReal(0.0), nil, err
    }
    g := make([]float64, x.Dim())
    for i := 0; i < x.Dim(); i++ {
      g[i] = y.GetDerivative(i)
    }
    return ConstReal(y.GetValue()), w, DenseConstRealVector(g), nil
  }
  return f_
}

/* -------------------------------------------------------------------------- */

type ProximalOperator func(x, w DenseBareRealVector, t *BareReal)

func ProxL1(lambda float64) ProximalOperator {
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    for i := 0; i < x.Dim(); i++ {
      if wi := w[i].GetValue(); wi < 0.0 {
        x[i].SetValue(-1.0*math.Max(math.Abs(wi) - lambda, 0.0))
      } else {
        x[i].SetValue( 1.0*math.Max(math.Abs(wi) - lambda, 0.0))
      }
    }
  }
  return f
}

func ProxL2(lambda float64) ProximalOperator {
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    t.Vnorm(w)
    t.Div(ConstReal(lambda), t)
    t.Sub(ConstReal(1.0), t)
    t.Max(ConstReal(0.0), t)
    x.VMULS(w, t)
  }
  return f
}

// Tikhonov regularization (1/2 * lambda * squared l2-norm)
func ProxTi(lambda float64) ProximalOperator {
  c := NewBareReal(1.0/(lambda + 1.0))
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    x.VMULS(w, c)
  }
  return f
}

/* -------------------------------------------------------------------------- */

func eval_stopping(xs, x1 DenseBareRealVector, epsilon float64) (bool, float64, error) {
  // evaluate stopping criterion
  max_x     := 0.0
  max_delta := 0.0
  delta     := 0.0
  for it := xs.JOINT_ITERATOR_(x1); it.Ok(); it.Next() {
    s1, s2 := it.GET()
    v1, v2 := 0.0, 0.0
    if s1 != nil {
      v1 = s1.GetValue()
    }
    if s2 != nil {
      v2 = s2.GetValue()
    }
    if math.IsNaN(v2) {
      return true, math.NaN(), fmt.Errorf("NaN value detected")
    }
    max_x     = math.Max(max_x    , math.Abs(v2))
    max_delta = math.Max(max_delta, math.Abs(v2 - v1))
  }
  if max_x != 0.0 {
    delta = max_delta/max_x
  } else {
    delta = max_delta
  }
  if max_x != 0.0 && max_delta/max_x <= epsilon ||
    (max_x == 0.0 && max_delta == 0.0) {
    return true, delta, nil
  }
  return false, delta, nil
}

/* -------------------------------------------------------------------------- */

func Run(f interface{}, n int, x Vector, args ...interface{}) (Vector, error) {

  hook          := Hook                  { nil}
  epsilon       := Epsilon               {1e-8}
  gamma         := Gamma                 {1.0/30.0}
  maxIterations := MaxIterations         {int(^uint(0) >> 1)}
  l1reg         := L1Regularization      { 0.0}
  l2reg         := L2Regularization      { 0.0}
  tireg         := TikhonovRegularization{ 0.0}
  proxop        := ProximalOperator      (nil)
  seed          := Seed                  {0}
  inSitu        := &InSitu               {}

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a
    case Epsilon:
      epsilon = a
    case Gamma:
      gamma = a
    case MaxIterations:
      maxIterations = a
    case L1Regularization:
      l1reg = a
    case L2Regularization:
      l2reg = a
    case TikhonovRegularization:
      tireg = a
    case ProximalOperator:
      proxop = a
    case Seed:
      seed = a
    case *InSitu:
      inSitu = a
    case InSitu:
      panic("InSitu must be passed by reference")
    default:
      panic("invalid optional argument")
    }
  }
  { m := 0
    if l1reg.Value != 0.0 { m++ }
    if l2reg.Value != 0.0 { m++ }
    if tireg.Value != 0.0 { m++ }
    if m > 1 {
      return x, fmt.Errorf("multiple regularizations are not supported")
    }
  }
  if l1reg.Value < 0.0 {
    return x, fmt.Errorf("invalid l1-regularization constant")
  }
  if l2reg.Value < 0.0 {
    return x, fmt.Errorf("invalid l2-regularization constant")
  }
  if tireg.Value < 0.0 {
    return x, fmt.Errorf("invalid l2-regularization constant")
  }
  if proxop == nil {
    switch {
    case l1reg.Value != 0.0: proxop = ProxL1(gamma.Value*l1reg.Value/float64(n))
    case l2reg.Value != 0.0: proxop = ProxL2(gamma.Value*l2reg.Value/float64(n))
    case tireg.Value != 0.0: proxop = ProxTi(gamma.Value*tireg.Value/float64(n))
    }
  }
  switch g := f.(type) {
  case Objective1Dense:
    return saga1Dense (g, n, x, gamma, epsilon, maxIterations, proxop, hook, seed, inSitu)
  case Objective2Dense:
    return saga2Dense (g, n, x, gamma, epsilon, maxIterations, proxop, hook, seed, inSitu)
  case Objective1Sparse:
    return saga1Sparse(g, n, x, gamma, epsilon, maxIterations, proxop, hook, seed, inSitu)
  case Objective2Sparse:
    return saga2Sparse(g, n, x, gamma, epsilon, maxIterations, proxop, hook, seed, inSitu)
  default:
    panic("invalid objective")
  }
}
