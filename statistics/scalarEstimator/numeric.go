/* Copyright (C) 2017 Philipp Benner
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

package scalarEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/bfgs"
import "github.com/pbenner/autodiff/algorithm/newton"
import "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NumericEstimator struct {
	ScalarPdf
	StdEstimator
	Method        string
	Epsilon       float64
	MaxIterations int
	StepInit      float64
	Eta           []float64
	Hook          func(variables ConstVector, r ConstScalar) error
}

/* -------------------------------------------------------------------------- */

func NewNumericEstimator(f ScalarPdf) (*NumericEstimator, error) {
	r := NumericEstimator{}
	r.ScalarPdf = f.CloneScalarPdf()
	r.Method = "newton"
	r.Epsilon = 1e-8
	r.MaxIterations = 20
	r.StepInit = 1e-2
	r.Eta = []float64{0.9, 1.1}
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) Clone() *NumericEstimator {
	r, _ := NewNumericEstimator(obj.ScalarPdf.CloneScalarPdf())
	r.Method = obj.Method
	r.Epsilon = obj.Epsilon
	r.MaxIterations = obj.MaxIterations
	r.Hook = obj.Hook
	r.StepInit = obj.StepInit
	r.Eta = []float64{obj.Eta[0], obj.Eta[1]}
	r.x = obj.x
	return r
}

func (obj *NumericEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) GetParameters() Vector {
	return obj.ScalarPdf.GetParameters()
}

func (obj *NumericEstimator) SetParameters(parameters Vector) error {
	return obj.ScalarPdf.SetParameters(parameters)
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	nt := p.NumberOfThreads()
	x := obj.x
	n := obj.n
	m := obj.x.Dim()
	// create a copy of the density function
	f := make([]ScalarPdf, nt)
	for i := 0; i < len(f); i++ {
		f[i] = obj.ScalarPdf.CloneScalarPdf()
	}
	constraints_f := func(variables Vector) bool {
		if err := f[0].SetParameters(variables); err != nil {
			return false
		}
		return true
	}
	// define the objective function
	objective_f := func(variables Vector) (Scalar, error) {
		// temporary variable
		t := NullVector(RealType, nt)
		s := NullVector(RealType, nt)
		r := NullVector(RealType, nt)
		for i := 0; i < len(f); i++ {
			if err := f[i].SetParameters(variables); err != nil {
				return nil, err
			}
		}
		g := p.NewJobGroup()
		p.AddRangeJob(0, m, g, func(k int, p ThreadPool, erf func() error) error {
			f := f[p.GetThreadId()]
			t := t.At(p.GetThreadId())
			s := s.At(p.GetThreadId())
			r := r.At(p.GetThreadId())
			// stop if there was an error in another thread
			if erf() != nil {
				return nil
			}
			if gamma != nil {
				if !math.IsInf(gamma.ConstAt(k).GetValue(), -1) {
					if err := f.LogPdf(t, x.ConstAt(k)); err != nil {
						return err
					}
					s.Exp(gamma.ConstAt(k))
					t.Mul(t, s)
					r.Add(r, t)
				}
			} else {
				if err := f.LogPdf(t, x.ConstAt(k)); err != nil {
					return err
				}
				r.Add(r, t)
			}
			return nil
		})
		p.Wait(g)
		// sum up results from all threads
		for i := 1; i < r.Dim(); i++ {
			r.At(0).Add(r.At(0), r.At(i))
		}
		if obj.Hook != nil {
			if err := obj.Hook(variables, r.At(0)); err != nil {
				return nil, err
			}
		}
		r.At(0).Neg(r.At(0))
		r.At(0).Div(r.At(0), NewReal(float64(n)))
		return r.At(0), nil
	}
	// get parameters of the density function and convert
	// the scalar type to real
	theta_0 := obj.ScalarPdf.GetParameters()
	theta_0 = AsVector(RealType, theta_0)

	var theta_n ConstVector
	var err error

	switch obj.Method {
	// execute optimization algorithm
	case "newton":
		theta_n, err = newton.RunMin(objective_f, theta_0,
			newton.Epsilon{obj.Epsilon},
			newton.MaxIterations{obj.MaxIterations},
			newton.Constraints{constraints_f},
			newton.HessianModification{"LDL"})
	case "bfgs":
		theta_n, err = bfgs.Run(objective_f, theta_0,
			bfgs.Epsilon{obj.Epsilon},
			bfgs.MaxIterations{obj.MaxIterations},
			bfgs.Constraints{constraints_f})
	case "rprop":
		theta_n, err = rprop.Run(objective_f, theta_0, obj.StepInit, obj.Eta,
			rprop.Epsilon{obj.Epsilon},
			rprop.Constraints{constraints_f})
	}
	if err != nil && err.Error() != "line search failed" {
		return err
	} else {
		// set parameters of the density function, but keep the
		// initial scalar type
		v := obj.GetParameters()
		v.Set(theta_n)
		obj.SetParameters(v)
	}
	return nil
}

func (obj *NumericEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *NumericEstimator) GetEstimate() (ScalarPdf, error) {
	return obj.ScalarPdf, nil
}
