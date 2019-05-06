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
type ObjectiveDense func(int, DenseBareRealVector) (ConstReal, ConstReal, DenseBareRealVector, bool, error)
type ProximalOperatorDense func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal)
type InSituDense struct {
  T1 DenseBareRealVector
  T2 *BareReal
}
/* -------------------------------------------------------------------------- */
type GradientDense struct {
  g DenseBareRealVector
  w ConstReal
  g_const bool
}
func (obj GradientDense) add(v DenseBareRealVector) {
  for it := v.JOINT_ITERATOR_(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.GET()
    if s_a == nil {
      s_a = v.AT(it.Index())
    }
    if s_b != nil {
      s_a.SetValue(s_a.GetValue() + obj.w.GetValue()*s_b.GetValue())
    }
  }
}
func (obj GradientDense) sub(v DenseBareRealVector) {
  for it := v.JOINT_ITERATOR_(obj.g); it.Ok(); it.Next() {
    s_a, s_b := it.GET()
    if s_a == nil {
      s_a = v.AT(it.Index())
    }
    if s_b != nil {
      s_a.SetValue(s_a.GetValue() - obj.w.GetValue()*s_b.GetValue())
    }
  }
}
func (obj *GradientDense) set(w ConstReal, g DenseBareRealVector, g_const bool) {
  if g_const {
    obj.g = g
  } else {
    if obj.g != nil {
      obj.g.SET(g)
    } else {
      obj.g = g.Clone()
    }
  }
  obj.g_const = g_const
  obj.w = w
}
/* -------------------------------------------------------------------------- */
func ProxL1Dense(lambda float64) ProximalOperatorDense {
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    for i := 0; i < x.Dim(); i++ {
      if yi := w.ValueAt(i); yi < 0.0 {
        x.AT(i).SetValue(-1.0*math.Max(math.Abs(yi) - lambda, 0.0))
      } else {
        x.AT(i).SetValue( 1.0*math.Max(math.Abs(yi) - lambda, 0.0))
      }
    }
  }
  return f
}
func ProxL2Dense(lambda float64) ProximalOperatorDense {
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    t.Vnorm(w)
    t.Div(ConstReal(lambda), t)
    t.Sub(ConstReal(1.0), t)
    t.Max(ConstReal(0.0), t)
    x.VMULS(w, t)
  }
  return f
}
// Tikhonov regularization (1/2 * squared l2-norm)
func ProxTiDense(lambda float64) ProximalOperatorDense {
  c := NewBareReal(1.0/(lambda + 1.0))
  f := func(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
    x.VMULS(w, c)
  }
  return f
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
  proxop ProximalOperatorDense,
  hook Hook,
  seed Seed,
  inSitu *InSituDense) (Vector, error) {
  x1 := AsDenseBareRealVector(x)
  x2 := AsDenseBareRealVector(x)
  // length of gradient
  d := x.Dim()
  // gradient
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
  // some constants
  t_n := NewBareReal(float64(n))
  t_g := NewBareReal(gamma.Value)
  // sum of gradients
  s := NullDenseBareRealVector(d)
  dict := make([]GradientDense, n)
  // initialize s and d
  for i := 0; i < n; i++ {
    if _, w, g, g_const, err := f(i, x1); err != nil {
      return nil, err
    } else {
      dict[i].set(w, g, g_const)
      dict[i].add(s)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))
  y := ConstReal(math.NaN())
  for i_ := 0; i_ < maxIterations.Value && i_/n < maxEpochs.Value; i_++ {
    j := g.Intn(n)
    // get old gradient
    g1 = dict[j]
    // evaluate objective function
    if y_, w, g, g_const, err := f(j, x1); err != nil {
      return x1, err
    } else {
      y = y_
      g2.set(w, g, g_const)
    }
    t1.VDIVS(s , t_n)
    g2.add(t1)
    g1.sub(t1)
    t1.VMULS(t1, t_g)
    if proxop != nil {
      t1.VSUBV(x1, t1)
      proxop(x2, t1, t2)
    } else {
      x2.VSUBV(x1, t1)
    }
    // evaluate stopping criterion
    max_x := 0.0
    max_delta := 0.0
    delta := 0.0
    for it := x1.JOINT_ITERATOR_(x2); it.Ok(); it.Next() {
      s1, s2 := it.GET()
      v1, v2 := 0.0, 0.0
      if s1 != nil {
        v1 = s1.GetValue()
      }
      if s2 != nil {
        v2 = s2.GetValue()
      }
      if math.IsNaN(v2) {
        return x1, fmt.Errorf("NaN value detected")
      }
      max_x = math.Max(max_x , math.Abs(v2))
      max_delta = math.Max(max_delta, math.Abs(v2 - v1))
    }
    if max_x != 0.0 {
      delta = max_delta/max_x
    } else {
      delta = max_delta
    }
    // execute hook if available
    if hook.Value != nil && hook.Value(x1, ConstReal(delta), y, i_) {
      break
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
    dict[j].set(g2.w, g2.g, g2.g_const)
  }
  return x1, nil
}
