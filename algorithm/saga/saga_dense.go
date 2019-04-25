/* -*- mode: go; -*-
 *
 * Copyright (C) 2019 Philipp Benner
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
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
import "fmt"
import "math"
import "math/rand"
import . "github.com/pbenner/autodiff"
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
type ObjectiveDense func(int, Vector) (ConstReal, DenseBareRealVector, ConstReal, error)
type InSituDense struct {
  T1 DenseBareRealVector
  T2 *BareReal
}
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
type GradientDense struct {
  g DenseBareRealVector
  w ConstReal
}
func (obj GradientDense) add(v DenseBareRealVector) {
  for it := v.JointIterator(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.Get()
    if s_a == nil {
      s_a = v.At(it.Index())
    }
    s_a.SetValue(s_a.GetValue() + obj.w.GetValue()*s_b.GetValue())
  }
}
func (obj GradientDense) sub(v DenseBareRealVector) {
  for it := v.JointIterator(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.Get()
    if s_a == nil {
      s_a = v.At(it.Index())
    }
    s_a.SetValue(s_a.GetValue() - obj.w.GetValue()*s_b.GetValue())
  }
}
func (obj *GradientDense) set(g DenseBareRealVector, w ConstReal) {
  if obj.g == nil {
    obj.g = NullDenseBareRealVector(g.Dim())
  }
  obj.g.Set(g)
  obj.w = w
}
/* -------------------------------------------------------------------------- */
func l1regularizationDense(x Vector, w DenseBareRealVector, t *BareReal, lambda float64) {
  for i := 0; i < x.Dim(); i++ {
    if yi := w.ValueAt(i); yi < 0.0 {
      x.At(i).SetValue(-1.0*math.Max(math.Abs(yi) - lambda, 0.0))
    } else {
      x.At(i).SetValue( 1.0*math.Max(math.Abs(yi) - lambda, 0.0))
    }
  }
}
func l2regularizationDense(x Vector, w DenseBareRealVector, t *BareReal, lambda float64) {
  t.Vnorm(w)
  t.Div(ConstReal(lambda), t)
  t.Sub(ConstReal(1.0), t)
  t.Max(ConstReal(0.0), t)
  x.VmulS(w, t)
}
/* -------------------------------------------------------------------------- */
func sagaDense(
  f ObjectiveDense,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxEpochs MaxEpochs,
  maxIterations MaxIterations,
  l1reg L1Regularization,
  l2reg L2Regularization,
  hook Hook,
  inSitu *InSituDense) (Vector, error) {
  x1 := x.CloneVector()
  x2 := x.CloneVector()
  // length of gradient
  d := x.Dim()
  // gradient
  var y ConstReal
  var g1 GradientDense
  var g2 GradientDense
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
  s := NullDenseBareRealVector(d)
  dict := make([]GradientDense, n)
  // initialize s and d
  for i := 0; i < n; i++ {
    if _, g, w, err := f(i, x1); err != nil {
      return nil, err
    } else {
      dict[i].set(g, w)
      dict[i].add(s)
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
    g2.add(t1)
    g1.sub(t1)
    t1.VmulS(t1, ConstReal(gamma.Value))
    switch {
    case l1reg.Value != 0.0:
      t1.VsubV(x1, t1)
      l1regularizationDense(x2, t1, t2, gamma.Value*l1reg.Value)
    case l2reg.Value != 0.0:
      t1.VsubV(x1, t1)
      l2regularizationDense(x2, t1, t2, gamma.Value*l2reg.Value)
    default:
      x2.VsubV(x1, t1)
    }
    // evaluate stopping criterion
    max_x := 0.0
    max_delta := 0.0
    for i := 0; i < d; i ++ {
      if math.IsNaN(x2.ValueAt(i)) {
        return x1, fmt.Errorf("NaN value detected")
      }
      max_x = math.Max(max_x , math.Abs(x2.ValueAt(i)))
      max_delta = math.Max(max_delta, math.Abs(x2.ValueAt(i) - x1.ValueAt(i)))
    }
    if max_x != 0.0 && max_delta/max_x <= epsilon.Value*gamma.Value ||
      (max_x == 0.0 && max_delta == 0.0) {
      return x2, nil
    }
    x1, x2 = x2, x1
    // update table
    g1.sub(s)
    g2.add(s)
    // update dictionary
    dict[j].set(g2.g, g2.w)
  }
  return x1, nil
}
