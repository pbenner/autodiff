/* -*- mode: go; -*-
 *
 * Copyright (C) 2019-2020 Philipp Benner
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
import "math/rand"
import . "github.com/pbenner/autodiff"
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
type Objective1Sparse func(int, DenseFloat64Vector) (float64, float64, SparseConstFloat64Vector, error)
type Objective2Sparse func(int, DenseFloat64Vector) (float64, SparseConstFloat64Vector, error)
/* -------------------------------------------------------------------------- */
type ConstGradientSparse struct {
  g SparseConstFloat64Vector
  w float64
}
func (obj ConstGradientSparse) update(g2 ConstGradientSparse, v DenseFloat64Vector) {
  c := g2.w - obj.w
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetFloat64(s_a.GetFloat64() + c*s_b.GetFloat64())
  }
}
func (obj ConstGradientSparse) add(v DenseFloat64Vector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetFloat64(s_a.GetFloat64() + obj.w*s_b.GetFloat64())
  }
}
func (obj *ConstGradientSparse) set(w float64, g SparseConstFloat64Vector) {
  obj.g = g
  obj.w = w
}
/* -------------------------------------------------------------------------- */
type GradientSparse struct {
  g DenseFloat64Vector
}
func (obj GradientSparse) add(v DenseFloat64Vector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetFloat64(s_a.GetFloat64() + s_b.GetFloat64())
  }
}
func (obj GradientSparse) sub(v DenseFloat64Vector) {
  for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
    s_a := v.AT(it.Index())
    s_b := it.GET()
    s_a.SetFloat64(s_a.GetFloat64() - s_b.GetFloat64())
  }
}
func (obj *GradientSparse) set(g ConstVector) {
  if obj.g != nil {
    obj.g.Set(g)
  } else {
    obj.g = AsDenseFloat64Vector(g)
  }
}
/* -------------------------------------------------------------------------- */
func saga1Sparse(
  f Objective1Sparse,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxIterations MaxIterations,
  proxop ProximalOperatorType,
  hook Hook,
  seed Seed,
  inSitu *InSitu) (Vector, int64, error) {
  xs := AsDenseFloat64Vector(x)
  x1 := AsDenseFloat64Vector(x)
  // length of gradient
  d := x.Dim()
  // gradient
  var g1 ConstGradientSparse
  var g2 ConstGradientSparse
  // allocate temporary memory
  if inSitu.T1 == nil {
    inSitu.T1 = NullDenseFloat64Vector(d)
  }
  // temporary variables
  t1 := inSitu.T1
  // some constants
  t_n := float64(n)
  t_g := gamma.Value
  // sum of gradients
  s := NullDenseFloat64Vector(d)
  // initialize s and d
  dict := make([]ConstGradientSparse, n)
  for i := 0; i < n; i++ {
    if _, w, gt, err := f(i, x1); err != nil {
      return nil, seed.Value, err
    } else {
      dict[i].set(w, gt)
      dict[i].add(s)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))
  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 0; i_ < n; i_++ {
      j := g.Intn(n)
      // get old gradient
      g1 = dict[j]
      // evaluate objective function
      if _, w, gt, err := f(j, x1); err != nil {
        return x1, g.Int63(), err
      } else {
        g2.set(w, gt)
      }
      gw1 := g1.w
      gw2 := g2.w
      c := gw2 - gw1
      if proxop == nil {
        for i := 0; i < s.Dim(); i++ {
          s_i := s.Float64At(i)
          g1i := g1.g.Float64At(i)
          x1i := x1.Float64At(i)
          x1.AT(i).SetFloat64(x1i - t_g*(c*g1i + s_i/t_n))
        }
      } else {
        for i := 0; i < s.Dim(); i++ {
          s_i := s.Float64At(i)
          g1i := g1.g.Float64At(i)
          x1i := x1.Float64At(i)
          t1.AT(i).SetFloat64(x1i - t_g*(c*g1i + s_i/t_n))
        }
        proxop.Eval(x1, t1)
      }
      // update gradient avarage
      g1.update(g2, s)
      // update dictionary
      dict[j].set(g2.w, g2.g)
    }
    if stop, delta, err := EvalStopping(xs, x1, epsilon.Value*gamma.Value); stop {
      return x1, g.Int63(), err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstFloat64(delta), ConstFloat64(float64(n)*proxop.GetLambda()/gamma.Value), epoch) {
        break
      }
    }
    xs.Set(x1)
  }
  return x1, g.Int63(), nil
}
func saga2Sparse(
  f Objective2Sparse,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxIterations MaxIterations,
  proxop ProximalOperatorType,
  hook Hook,
  seed Seed,
  inSitu *InSitu) (Vector, int64, error) {
  xs := AsDenseFloat64Vector(x)
  x1 := AsDenseFloat64Vector(x)
  // length of gradient
  d := x.Dim()
  // gradient
  var g1 GradientSparse
  var g2 GradientSparse
  // allocate temporary memory
  if inSitu.T1 == nil {
    inSitu.T1 = NullDenseFloat64Vector(d)
  }
  // temporary variables
  t1 := inSitu.T1
  // some constants
  t_n := float64(n)
  t_g := gamma.Value
  // sum of gradients
  s := NullDenseFloat64Vector(d)
  // initialize s and d
  dict := make([]GradientSparse, n)
  for i := 0; i < n; i++ {
    if _, gt, err := f(i, x1); err != nil {
      return nil, seed.Value, err
    } else {
      dict[i].set(gt)
      dict[i].add(s)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))
  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 0; i_ < n; i_++ {
      j := g.Intn(n)
      // get old gradient
      g1 = dict[j]
      // evaluate objective function
      if _, gt, err := f(j, x1); err != nil {
        return x1, g.Int63(), err
      } else {
        g2.set(gt)
      }
      if proxop == nil {
        for i := 0; i < s.Dim(); i++ {
          s_i := s.Float64At(i)
          g1i := g1.g.Float64At(i)
          g2i := g2.g.Float64At(i)
          x1i := x1.Float64At(i)
          x1.AT(i).SetFloat64(x1i - t_g*(g2i - g1i + s_i/t_n))
        }
      } else {
        for i := 0; i < s.Dim(); i++ {
          s_i := s.Float64At(i)
          g1i := g1.g.Float64At(i)
          g2i := g2.g.Float64At(i)
          x1i := x1.Float64At(i)
          t1.AT(i).SetFloat64(x1i - t_g*(g2i - g1i + s_i/t_n))
        }
        proxop.Eval(x1, t1)
      }
      // update gradient avarage
      g1.sub(s)
      g2.add(s)
      // update dictionary
      dict[j].set(g2.g)
    }
    if stop, delta, err := EvalStopping(xs, x1, epsilon.Value*gamma.Value); stop {
      return x1, g.Int63(), err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstFloat64(delta), ConstFloat64(float64(n)*proxop.GetLambda()/gamma.Value), epoch) {
        break
      }
    }
    xs.Set(x1)
  }
  return x1, g.Int63(), nil
}
