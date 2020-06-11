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
import "math/rand"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type GradientJit struct {
	G SparseConstRealVector
	W ConstReal
}

func (obj GradientJit) Update(g2 GradientJit, v DenseBareRealVector) {
	g := obj.G.GetSparseValues()
	c := g2.W - obj.W
	for i, k := range obj.G.GetSparseIndices() {
		v[k] = v[k] + BareReal(c.GetValue()*g[i])
	}
}

func (obj GradientJit) Add(v DenseBareRealVector) {
	g := obj.G.GetSparseValues()
	for i, k := range obj.G.GetSparseIndices() {
		v[k] = v[k] + BareReal(obj.W.GetValue()*g[i])
	}
}

func (obj *GradientJit) Set(w ConstReal, g SparseConstRealVector) {
	obj.G = g
	obj.W = w
}

/* -------------------------------------------------------------------------- */

func sagaJit(
	f Objective1Sparse,
	n int,
	x Vector,
	gamma Gamma,
	epsilon Epsilon,
	maxIterations MaxIterations,
	jit JitUpdateType,
	hook Hook,
	seed Seed,
	inSitu *InSitu) (Vector, int64, error) {

	xs := AsDenseBareRealVector(x)
	x1 := AsDenseBareRealVector(x)
	xk := make([]int, x.Dim())

	// length of gradient
	d := x.Dim()
	// gradient
	var g1 GradientJit
	var g2 GradientJit

	// some constants
	t_n := BareReal(n)
	t_g := BareReal(gamma.Value)

	// sum of gradients
	s := NullDenseBareRealVector(d)
	// initialize s and d
	dict := make([]GradientJit, n)
	for i := 0; i < n; i++ {
		if _, w, gt, err := f(i, x1); err != nil {
			return nil, seed.Value, err
		} else {
			dict[i].Set(w, gt)
			dict[i].Add(s)
		}
	}
	g := rand.New(rand.NewSource(seed.Value))

	for epoch := 0; epoch < maxIterations.Value; epoch++ {
		for i_ := 0; i_ < n; i_++ {
			j := g.Intn(n)
			// get old gradient
			g1 = dict[j]
			// perform jit updates for all x_i where g_i != 0
			for _, k := range g1.G.GetSparseIndices() {
				if m := i_ - xk[k]; m > 0 {
					x1[k] = jit.Update(x1[k], t_g*s[k]/t_n, k, m)
				}
			}
			// evaluate objective function
			if _, w, gt, err := f(j, x1); err != nil {
				return x1, g.Int63(), err
			} else {
				g2.Set(w, gt)
			}
			c := BareReal(g2.W - g1.W)
			v := g1.G.GetSparseValues()
			for i, k := range g1.G.GetSparseIndices() {
				x1[k] = x1[k] - t_g*(1.0-1.0/t_n)*c*BareReal(v[i])
				xk[k] = i_
			}
			// update gradient avarage
			g1.Update(g2, s)

			// update dictionary
			dict[j].Set(g2.W, g2.G)
		}
		// compute missing updates of x1
		for k := 0; k < x1.Dim(); k++ {
			if m := n - xk[k]; m > 0 {
				x1[k] = jit.Update(x1[k], t_g*s[k]/t_n, k, m)
			}
			// reset xk
			xk[k] = 0
		}
		if stop, delta, err := EvalStopping(xs, x1, epsilon.Value*gamma.Value); stop {
			return x1, g.Int63(), err
		} else {
			// execute hook if available
			if hook.Value != nil && hook.Value(x1, ConstReal(delta), ConstReal(float64(n)*jit.GetLambda()/gamma.Value), epoch) {
				break
			}
		}
		xs.SET(x1)
	}
	return x1, g.Int63(), nil
}
