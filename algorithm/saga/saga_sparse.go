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
//import   "fmt"
import "math"
import "math/rand"
import . "github.com/pbenner/autodiff"
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
type Objective1Sparse func(int, DenseBareRealVector) (ConstReal, ConstReal, SparseConstRealVector, error)
type Objective2Sparse func(int, DenseBareRealVector) (ConstReal, SparseConstRealVector, error)
type InSituSparse struct {
  T1 DenseBareRealVector
  T2 *BareReal
}
/* -------------------------------------------------------------------------- */
type ConstGradientSparse struct {
  g SparseConstRealVector
  w ConstReal
}
func (obj ConstGradientSparse) add(v DenseBareRealVector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetValue(s_a.GetValue() + obj.w.GetValue()*s_b.GetValue())
  }
}
func (obj ConstGradientSparse) sub(v DenseBareRealVector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetValue(s_a.GetValue() - obj.w.GetValue()*s_b.GetValue())
  }
}
func (obj *ConstGradientSparse) set(w ConstReal, g SparseConstRealVector) {
  obj.g = g
  obj.w = w
}
/* -------------------------------------------------------------------------- */
type GradientSparse struct {
  g DenseBareRealVector
}
func (obj GradientSparse) add(v DenseBareRealVector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetValue(s_a.GetValue() + s_b.GetValue())
  }
}
func (obj GradientSparse) sub(v DenseBareRealVector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetValue(s_a.GetValue() - s_b.GetValue())
  }
}
func (obj *GradientSparse) set(g ConstVector) {
  if obj.g != nil {
    obj.g.Set(g)
  } else {
    obj.g = AsDenseBareRealVector(g)
  }
}
/* -------------------------------------------------------------------------- */
func sagaSparse(
  f1 Objective1Sparse,
  f2 Objective2Sparse,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxIterations MaxIterations,
  proxop ProximalOperator,
  hook Hook,
  seed Seed,
  inSitu *InSituSparse) (Vector, error) {
  xs := AsDenseBareRealVector(x)
  x1 := AsDenseBareRealVector(x)
  x2 := AsDenseBareRealVector(x)
  // length of gradient
  d := x.Dim()
  // gradient
  var g1 GradientSparse
  var g2 GradientSparse
  var g1_const ConstGradientSparse
  var g2_const ConstGradientSparse
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
  t_n := float64(n)
  t_g := gamma.Value
  // sum of gradients
  s := NullDenseBareRealVector(d)
  var dict [] GradientSparse
  var dict_const []ConstGradientSparse
  // initialize s and d
  if f1 != nil {
    dict_const = make([]ConstGradientSparse, n)
    for i := 0; i < n; i++ {
      if _, w, g, err := f1(i, x1); err != nil {
        return nil, err
      } else {
        dict_const[i].set(w, g)
        dict_const[i].add(s)
      }
    }
  } else {
    dict := make([]GradientSparse, n)
    for i := 0; i < n; i++ {
      if _, g, err := f2(i, x1); err != nil {
        return nil, err
      } else {
        dict[i].set(g)
        dict[i].add(s)
      }
    }
  }
  g := rand.New(rand.NewSource(seed.Value))
  y := ConstReal(math.NaN())
  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    if f1 != nil {
      for i_ := 0; i_ < n; i_++ {
        j := g.Intn(n)
        // get old gradient
        g1_const = dict_const[j]
        // evaluate objective function
        if y_, w, g, err := f1(j, x1); err != nil {
          return x1, err
        } else {
          y = y_
          g2_const.set(w, g)
        }
        gw1 := g1_const.w.GetValue()
        gw2 := g2_const.w.GetValue()
        c := gw2 - gw1
        for i := 0; i < s.Dim(); i++ {
          s_i := s.ValueAt(i)
          g1i := g1_const.g.ValueAt(i)
          t1.AT(i).SetValue(t_g*(c*g1i + s_i/t_n))
        }
        if proxop != nil {
          t1.VSUBV(x1, t1)
          proxop(x2, t1, t2)
        } else {
          x2.VSUBV(x1, t1)
        }
        x1, x2 = x2, x1
        // update gradient avarage
        g1_const.sub(s)
        g2_const.add(s)
        // update dictionary
        dict_const[j].set(g2_const.w, g2_const.g)
      }
    } else {
      for i_ := 0; i_ < n; i_++ {
        j := g.Intn(n)
        // get old gradient
        g1 = dict[j]
        // evaluate objective function
        if y_, g, err := f2(j, x1); err != nil {
          return x1, err
        } else {
          y = y_
          g2.set(g)
        }
        for i := 0; i < s.Dim(); i++ {
          s_i := s.ValueAt(i)
          g1i := g1.g.ValueAt(i)
          g2i := g2.g.ValueAt(i)
          t1.AT(i).SetValue(t_g*(g2i - g1i + s_i/t_n))
        }
        if proxop != nil {
          t1.VSUBV(x1, t1)
          proxop(x2, t1, t2)
        } else {
          x2.VSUBV(x1, t1)
        }
        x1, x2 = x2, x1
        // update gradient avarage
        g1.sub(s)
        g2.add(s)
        // update dictionary
        dict[j].set(g2.g)
      }
    }
    if stop, delta, err := eval_stopping(xs, x1, epsilon.Value*gamma.Value); stop {
      return x1, err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstReal(delta), y, epoch) {
        break
      }
    }
    xs.Set(x1)
  }
  return x1, nil
}
