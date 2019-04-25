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

// f_i(x) -> (y, gradient_vector, gradient_weight, error)
type objective func(int, Vector) (ConstScalar, ConstVector, ConstScalar, error)

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

type InSitu struct {
  T1 Vector
  T2 Scalar
}

/* -------------------------------------------------------------------------- */

func Wrapper(f func(int, Vector, Scalar) error) objective {
  y := NullReal()
  w := ConstReal(1.0)
  g := func(i int, x Vector) (ConstScalar, ConstVector, ConstScalar, error) {
    x.Variables(1)
    if err := f(i, x, y); err != nil {
      return nil, nil, nil, err
    }
    g := DenseGradient{y}
    return ConstReal(y.GetValue()), g, w, nil
  }
  return g
}

/* -------------------------------------------------------------------------- */

type gradientType struct {
  g DenseBareRealVector
  w *BareReal
}

func (obj gradientType) Vadd(v Vector) {
  for it := v.JointIterator(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.Get()
    if s_a == nil {
      s_a = v.At(it.Index())
    }
    s_a.SetValue(s_a.GetValue() + obj.w.GetValue()*s_b.GetValue())
  }
}

func (obj gradientType) Vsub(v Vector) {
  for it := v.JointIterator(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.Get()
    if s_a == nil {
      s_a = v.At(it.Index())
    }
    s_a.SetValue(s_a.GetValue() - obj.w.GetValue()*s_b.GetValue())
  }
}

func (obj *gradientType) set(g ConstVector, w ConstScalar) {
  if obj.g == nil {
    obj.g = NullDenseBareRealVector(g.Dim())
  }
  if obj.w == nil {
    obj.w = NullBareReal()
  }
  obj.g.Set(g)
  obj.w.Set(w)
}

/* -------------------------------------------------------------------------- */

func l1regularization(x Vector, w ConstVector, t Scalar, lambda float64) {
  for i := 0; i < x.Dim(); i++ {
    if yi := w.ValueAt(i); yi < 0.0 {
      x.At(i).SetValue(-1.0*math.Max(math.Abs(yi) - lambda, 0.0))
    } else {
      x.At(i).SetValue( 1.0*math.Max(math.Abs(yi) - lambda, 0.0))
    }
  }
}

func l2regularization(x Vector, w ConstVector, t Scalar, lambda float64) {
  t.Vnorm(w)
  t.Div(ConstReal(lambda), t)
  t.Sub(ConstReal(1.0), t)
  t.Max(ConstReal(0.0), t)
  x.VmulS(w, t)
}

/* -------------------------------------------------------------------------- */

func saga(
  f objective,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxEpochs MaxEpochs,
  maxIterations MaxIterations,
  l1reg L1Regularization,
  l2reg L2Regularization,
  hook Hook,
  inSitu *InSitu) (Vector, error) {

  x1 := x.CloneVector()
  x2 := x.CloneVector()

  // length of gradient
  d := x.Dim()
  // gradient
  var y  ConstScalar
  var g1 gradientType
  var g2 gradientType

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
  dict := make([]gradientType, n)
  // initialize s and d
  for i := 0; i < n; i++ {
    if _, g, w, err := f(i, x1); err != nil {
      return nil, err
    } else {
      dict[i].set(g, w)
      dict[i].Vadd(s)
    }
  }
  y = ConstReal(math.NaN())

  for i_ := 0; i_ < maxIterations.Value && i_/n < maxEpochs.Value; i_++ {
    // execute hook if available
    if hook.Value != nil && hook.Value(x1, s, y) {
      break
    }
    j := rand.Intn(n)

    // get old gradient
    g1 = dict[j]
    // evaluate objective function
    if y_, g, w, err := f(j, x1); err != nil {
      return x1, err
    } else {
      y = y_
      g2.set(g, w)
    }

    t1.VdivS(s , ConstReal(float64(n)))
    g2.Vadd(t1)
    g1.Vsub(t1)
    t1.VmulS(t1, ConstReal(gamma.Value))

    switch {
    case l1reg.Value != 0.0:
      t1.VsubV(x1, t1)
      l1regularization(x2, t1, t2, gamma.Value*l1reg.Value)
    case l2reg.Value != 0.0:
      t1.VsubV(x1, t1)
      l2regularization(x2, t1, t2, gamma.Value*l2reg.Value)
    default:
      x2.VsubV(x1, t1)
    }
    // evaluate stopping criterion
    max_x     := 0.0
    max_delta := 0.0
    for i := 0; i < d; i ++ {
      if math.IsNaN(x2.ValueAt(i)) {
        return x1, fmt.Errorf("NaN value detected")
      }
      max_x     = math.Max(max_x    , math.Abs(x2.ValueAt(i)))
      max_delta = math.Max(max_delta, math.Abs(x2.ValueAt(i) - x1.ValueAt(i)))
    }
    if max_x != 0.0 && max_delta/max_x <= epsilon.Value*gamma.Value ||
      (max_x == 0.0 && max_delta == 0.0) {
      return x2, nil
    }
    x1, x2 = x2, x1
    // update table
    g1.Vsub(s)
    g2.Vadd(s)

    // update dictionary
    dict[j].set(g2.g, g2.w)
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

func run(f objective, n int, x Vector, args ...interface{}) (Vector, error) {

  hook          := Hook               {   nil}
  epsilon       := Epsilon            {  1e-8}
  gamma         := Gamma              {1.0/30.0}
  maxEpochs     := MaxEpochs          {int(^uint(0) >> 1)}
  maxIterations := MaxIterations      {int(^uint(0) >> 1)}
  l1reg         := L1Regularization   { 0.0}
  l2reg         := L2Regularization   { 0.0}
  inSitu        := &InSitu            {}

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
    case *InSitu:
      inSitu = a
    case InSitu:
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

  return saga(f, n, x, gamma, epsilon, maxEpochs, maxIterations, l1reg, l2reg, hook, inSitu)
}

/* -------------------------------------------------------------------------- */

func Run(f objective, n int, x Vector, args ...interface{}) (Vector, error) {

  return run(f, n, x, args...)
}
