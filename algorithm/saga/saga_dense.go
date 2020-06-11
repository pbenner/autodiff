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
import "math/rand"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
type Objective1Dense func(int, DenseBareRealVector) (ConstReal, ConstReal, DenseConstRealVector, error)
type Objective2Dense func(int, DenseBareRealVector) (ConstReal, DenseConstRealVector, error)

/* -------------------------------------------------------------------------- */
type ConstGradientDense struct {
	g DenseConstRealVector
	w ConstReal
}

func (obj ConstGradientDense) update(g2 ConstGradientDense, v DenseBareRealVector) {
	c := g2.w.GetValue() - obj.w.GetValue()
	for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
		s_a := v.AT(it.Index())
		s_b := it.GET()
		s_a.SetValue(s_a.GetValue() + c*s_b.GetValue())
	}
}
func (obj ConstGradientDense) add(v DenseBareRealVector) {
	for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
		s_a := v.AT(it.Index())
		s_b := it.GET()
		s_a.SetValue(s_a.GetValue() + obj.w.GetValue()*s_b.GetValue())
	}
}
func (obj *ConstGradientDense) set(w ConstReal, g DenseConstRealVector) {
	obj.g = g
	obj.w = w
}

/* -------------------------------------------------------------------------- */
type GradientDense struct {
	g DenseBareRealVector
}

func (obj GradientDense) add(v DenseBareRealVector) {
	for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
		s_a := v.AT(it.Index())
		s_b := it.GET()
		s_a.SetValue(s_a.GetValue() + s_b.GetValue())
	}
}
func (obj GradientDense) sub(v DenseBareRealVector) {
	for it := obj.g.ITERATOR(); it.Ok(); it.Next() {
		s_a := v.AT(it.Index())
		s_b := it.GET()
		s_a.SetValue(s_a.GetValue() - s_b.GetValue())
	}
}
func (obj *GradientDense) set(g ConstVector) {
	if obj.g != nil {
		obj.g.Set(g)
	} else {
		obj.g = AsDenseBareRealVector(g)
	}
}

/* -------------------------------------------------------------------------- */
func saga1Dense(
	f Objective1Dense,
	n int,
	x Vector,
	gamma Gamma,
	epsilon Epsilon,
	maxIterations MaxIterations,
	proxop ProximalOperatorType,
	hook Hook,
	seed Seed,
	inSitu *InSitu) (Vector, int64, error) {
	xs := AsDenseBareRealVector(x)
	x1 := AsDenseBareRealVector(x)
	// length of gradient
	d := x.Dim()
	// gradient
	var g1 ConstGradientDense
	var g2 ConstGradientDense
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
	// initialize s and d
	dict := make([]ConstGradientDense, n)
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
			gw1 := g1.w.GetValue()
			gw2 := g2.w.GetValue()
			c := gw2 - gw1
			if proxop == nil {
				for i := 0; i < s.Dim(); i++ {
					s_i := s.ValueAt(i)
					g1i := g1.g.ValueAt(i)
					x1i := x1.ValueAt(i)
					x1.AT(i).SetValue(x1i - t_g*(c*g1i+s_i/t_n))
				}
			} else {
				for i := 0; i < s.Dim(); i++ {
					s_i := s.ValueAt(i)
					g1i := g1.g.ValueAt(i)
					x1i := x1.ValueAt(i)
					t1.AT(i).SetValue(x1i - t_g*(c*g1i+s_i/t_n))
				}
				proxop.Eval(x1, t1, t2)
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
			if hook.Value != nil && hook.Value(x1, ConstReal(delta), ConstReal(float64(n)*proxop.GetLambda()/gamma.Value), epoch) {
				break
			}
		}
		xs.SET(x1)
	}
	return x1, g.Int63(), nil
}
func saga2Dense(
	f Objective2Dense,
	n int,
	x Vector,
	gamma Gamma,
	epsilon Epsilon,
	maxIterations MaxIterations,
	proxop ProximalOperatorType,
	hook Hook,
	seed Seed,
	inSitu *InSitu) (Vector, int64, error) {
	xs := AsDenseBareRealVector(x)
	x1 := AsDenseBareRealVector(x)
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
	t_n := float64(n)
	t_g := gamma.Value
	// sum of gradients
	s := NullDenseBareRealVector(d)
	// initialize s and d
	dict := make([]GradientDense, n)
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
					s_i := s.ValueAt(i)
					g1i := g1.g.ValueAt(i)
					g2i := g2.g.ValueAt(i)
					x1i := x1.ValueAt(i)
					x1.AT(i).SetValue(x1i - t_g*(g2i-g1i+s_i/t_n))
				}
			} else {
				for i := 0; i < s.Dim(); i++ {
					s_i := s.ValueAt(i)
					g1i := g1.g.ValueAt(i)
					g2i := g2.g.ValueAt(i)
					x1i := x1.ValueAt(i)
					t1.AT(i).SetValue(x1i - t_g*(g2i-g1i+s_i/t_n))
				}
				proxop.Eval(x1, t1, t2)
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
			if hook.Value != nil && hook.Value(x1, ConstReal(delta), ConstReal(float64(n)*proxop.GetLambda()/gamma.Value), epoch) {
				break
			}
		}
		xs.SET(x1)
	}
	return x1, g.Int63(), nil
}
