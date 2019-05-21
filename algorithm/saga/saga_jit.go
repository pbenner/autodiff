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

//import   "fmt"
import   "math/rand"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func sagaJit(
  f Objective1Sparse,
  n int,
  x Vector,
  gamma Gamma,
  epsilon Epsilon,
  maxIterations MaxIterations,
  proxop ProximalOperatorJitType,
  hook Hook,
  seed Seed,
  inSitu *InSitu) (Vector, error) {

  xs := AsDenseBareRealVector(x)
  x1 := AsDenseBareRealVector(x)
  xk := make([]int, x.Dim())

  // length of gradient
  d := x.Dim()
  // gradient
  var g1 ConstGradientSparse
  var g2 ConstGradientSparse

  // allocate temporary memory
  if inSitu.T1 == nil {
    inSitu.T1 = NullDenseBareRealVector(d)
  }
  // temporary variables
  t1 := BareReal(0.0)
  t2 := BareReal(0.0)
  // some constants
  t_n := BareReal(n)
  t_g := BareReal(gamma.Value)

  // sum of gradients
  s := NullDenseBareRealVector(d)
  // initialize s and d
  dict := make([]ConstGradientSparse, n)
  for i := 0; i < n; i++ {
    if _, w, g, err := f(i, x1); err != nil {
      return nil, err
    } else {
      dict[i].set(w, g)
      dict[i].add(s)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))

  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 1; i_ < n+1; i_++ {
      j := g.Intn(n)

      // get old gradient
      g1 = dict[j]
      // perform jit updates for all x_i where g_i != 0
      for _, k := range g1.g.GetSparseIndices() {
        if m := i_ - xk[k]; m > 1 {
          t1 = x1[k] - BareReal(m-1)*t_g*s[k]/t_n
          proxop.Eval(&x1[k], &t1, k, m-1, &t2)
        }
      }
      // evaluate objective function
      if _, w, g, err := f(j, x1); err != nil {
        return x1, err
      } else {
        g2.set(w, g)
      }
      c := BareReal(g2.w - g1.w)
      v := g1.g.GetSparseValues()
      for i, k := range g1.g.GetSparseIndices() {
        t1 = x1[k] - t_g*(c*BareReal(v[i]) + s[k]/t_n)
        proxop.Eval(&x1[k], &t1, k, 1, &t2)
        xk[k] = i_
      }
      // update gradient avarage
      g1.update(g2, s)

      // update dictionary
      dict[j].set(g2.w, g2.g)
    }
    // compute missing updates of x1
    for k := 0; k < x1.Dim(); k++ {
      if m := n - xk[k]; m > 0 {
        t1 = x1[k] - BareReal(m)*t_g*s[k]/t_n
        proxop.Eval(&x1[k], &t1, k, m, &t2)
      }
      // reset xk
      xk[k] = 0
    }
    if stop, delta, err := eval_stopping(xs, x1, epsilon.Value*gamma.Value); stop {
      return x1, err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstReal(delta), epoch) {
        break
      }
    }
    xs.SET(x1)
  }
  return x1, nil
}
