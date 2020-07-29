/* Copyright (C) 2019-2020 Philipp Benner
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

type ProximalOperator struct {
  Value ProximalOperatorType
}

type JitUpdate struct {
  Value JitUpdateType
}

type Hook struct {
  Value func(ConstVector, ConstScalar, ConstScalar, int) bool
}

type MaxIterations struct {
  Value int
}

type Seed struct {
  Value int64
}

type InSitu struct {
  T1 DenseFloat64Vector
}

/* -------------------------------------------------------------------------- */

func WrapperDense(f func(int, Vector, MagicScalar) error) Objective2Dense {
  x := NullDenseReal64Vector(0)
  y := NullReal64()
  f_ := func(i int, x_ DenseFloat64Vector) (float64, DenseFloat64Vector, error) {
    if x.Dim() == 0 {
      x = NullDenseReal64Vector(x_.Dim())
    }
    x.Set(x_)
    x.Variables(1)
    if err := f(i, x, y); err != nil {
      return 0.0, nil, err
    }
    g := make([]float64, x.Dim())
    for i := 0; i < x.Dim(); i++ {
      g[i] = y.GetDerivative(i)
    }
    return y.GetFloat64(), DenseFloat64Vector(g), nil
  }
  return f_
}

/* -------------------------------------------------------------------------- */

type ProximalOperatorType interface {
  GetLambda(       ) float64
  SetLambda(float64)
  Eval     (x, w DenseFloat64Vector)
}

type JitUpdateType interface {
  GetLambda(       ) float64
  SetLambda(float64)
  Update   (x, y float64, k, n int) float64
}

/* -------------------------------------------------------------------------- */

type ProximalOperatorL1 struct {
  Lambda float64
}

func (obj *ProximalOperatorL1) GetLambda() float64 {
  return obj.Lambda
}

func (obj *ProximalOperatorL1) SetLambda(lambda float64) {
  obj.Lambda = lambda
}

func (obj *ProximalOperatorL1) Eval(x DenseFloat64Vector, w DenseFloat64Vector) {
  for i := 0; i < x.Dim(); i++ {
    // sign(wi)*max{|wi| - n*lambda}
    if wi := w[i]; wi < 0.0 {
      if -wi < obj.Lambda {
        x[i] = 0.0
      } else {
        x[i] = wi + obj.Lambda
      }
    } else {
      if  wi < obj.Lambda {
        x[i] = 0.0
      } else {
        x[i] = wi - obj.Lambda
      }
    }
  }
}

/* -------------------------------------------------------------------------- */

type ProximalOperatorL2 struct {
  Lambda float64
}

func (obj *ProximalOperatorL2) GetLambda() float64 {
  return obj.Lambda
}

func (obj *ProximalOperatorL2) SetLambda(lambda float64) {
  obj.Lambda = lambda
}

func (obj *ProximalOperatorL2) Eval(x DenseFloat64Vector, w DenseFloat64Vector) {
  t := NullFloat64()
  t.Vnorm(w)
  t.Div(ConstFloat64(obj.Lambda), t)
  t.Sub(ConstFloat64(1.0), t)
  t.Max(ConstFloat64(0.0), t)
  x.VmulS(w, t)
}

/* -------------------------------------------------------------------------- */

type ProximalOperatorTi struct {
  Lambda float64
}

func (obj *ProximalOperatorTi) GetLambda() float64 {
  return obj.Lambda
}

func (obj *ProximalOperatorTi) SetLambda(lambda float64) {
  obj.Lambda = lambda
}

func (obj *ProximalOperatorTi) Eval(x DenseFloat64Vector, w DenseFloat64Vector) {
  x.VmulS(w, ConstFloat64(1.0/(obj.Lambda + 1.0)))
}

/* -------------------------------------------------------------------------- */

type JitUpdateL1 struct {
  Lambda float64
}

func (obj *JitUpdateL1) GetLambda() float64 {
  return obj.Lambda
}

func (obj *JitUpdateL1) SetLambda(lambda float64) {
  obj.Lambda = lambda
}

func (obj *JitUpdateL1) update(w float64, k, n int) float64 {
  // sign(w)*max{|w| - n*lambda}
  if w < 0.0 {
    if l := float64(n)*obj.Lambda; -w < l {
      return 0.0
    } else {
      return w + l
    }
  } else {
    if l := float64(n)*obj.Lambda;  w < l {
      return 0.0
    } else {
      return w - l
    }
  }
}

func (obj *JitUpdateL1) Update(x, y float64, k, m int) float64 {
  if y < obj.GetLambda() {
    w := x - float64(m)*y
    x  = obj.update(w, k, m)
  } else {
    for j := 0; j < m; j++ {
      w := x - y
      x  = obj.update(w, k, 1)
    }
  }
  return x
}

/* -------------------------------------------------------------------------- */

func EvalStopping(xs, x1 DenseFloat64Vector, epsilon float64) (bool, float64, error) {
  // evaluate stopping criterion
  max_x     := 0.0
  max_delta := 0.0
  delta     := 0.0
  for it := xs.JOINT_ITERATOR(x1); it.Ok(); it.Next() {
    s1, s2 := it.Get()
    v1, v2 := 0.0, 0.0
    if s1 != nil {
      v1 = s1.GetFloat64()
    }
    if s2 != nil {
      v2 = s2.GetFloat64()
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

func Run(f interface{}, n int, x Vector, args ...interface{}) (Vector, int64, error) {

  hook          := Hook                  { nil}
  epsilon       := Epsilon               {1e-8}
  gamma         := Gamma                 {1.0/30.0}
  maxIterations := MaxIterations         {int(^uint(0) >> 1)}
  l1reg         := L1Regularization      { 0.0}
  l2reg         := L2Regularization      { 0.0}
  tireg         := TikhonovRegularization{ 0.0}
  proxop        := ProximalOperator      {}
  jitUpdate     := JitUpdate             {}
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
    case JitUpdate:
      jitUpdate = a
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
      return x, seed.Value, fmt.Errorf("multiple regularizations are not supported")
    }
  }
  if l1reg.Value < 0.0 {
    return x, seed.Value, fmt.Errorf("invalid l1-regularization constant")
  }
  if l2reg.Value < 0.0 {
    return x, seed.Value, fmt.Errorf("invalid l2-regularization constant")
  }
  if tireg.Value < 0.0 {
    return x, seed.Value, fmt.Errorf("invalid ti-regularization constant")
  }
  // initialize proximal operator
  switch {
  case l1reg.Value != 0.0: proxop.Value = &ProximalOperatorL1{l1reg.Value}
  case l2reg.Value != 0.0: proxop.Value = &ProximalOperatorL2{l2reg.Value}
  case tireg.Value != 0.0: proxop.Value = &ProximalOperatorTi{tireg.Value}
  }
  // check arguments
  if proxop.Value != nil && jitUpdate.Value != nil {
    return x, seed.Value, fmt.Errorf("invalid arguments")
  }
  // rescale lambda
  if proxop.Value != nil {
    proxop.Value.SetLambda(gamma.Value*proxop.Value.GetLambda()/float64(n))
  }
  if jitUpdate.Value != nil {
    jitUpdate.Value.SetLambda(gamma.Value*jitUpdate.Value.GetLambda()/float64(n))
  }
  if jitUpdate.Value != nil {
    switch g := f.(type) {
    case Objective1Sparse:
      return sagaJit(g, n, x, gamma, epsilon, maxIterations, jitUpdate.Value, hook, seed, inSitu)
    default:
      panic("invalid objective")
    }
  } else {
    switch g := f.(type) {
    case Objective1Dense:
      return saga1Dense (g, n, x, gamma, epsilon, maxIterations, proxop.Value, hook, seed, inSitu)
    case Objective2Dense:
      return saga2Dense (g, n, x, gamma, epsilon, maxIterations, proxop.Value, hook, seed, inSitu)
    case Objective1Sparse:
      return saga1Sparse(g, n, x, gamma, epsilon, maxIterations, proxop.Value, hook, seed, inSitu)
    case Objective2Sparse:
      return saga2Sparse(g, n, x, gamma, epsilon, maxIterations, proxop.Value, hook, seed, inSitu)
    default:
      panic("invalid objective")
    }
  }
}
