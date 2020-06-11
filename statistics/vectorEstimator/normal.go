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

package vectorEstimator

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NormalEstimator struct {
	*vectorDistribution.NormalDistribution
	StdEstimator
	// parameters
	n        int
	SigmaMin float64
	// state
	sum_g     []float64
	sum_m     [][]float64
	sum_s     [][][]float64
	gamma_max float64
}

/* -------------------------------------------------------------------------- */

func NewNormalEstimator(mu, sigma []float64, sigmaMin float64) (*NormalEstimator, error) {
	if n := len(mu); n == 0 {
		return nil, fmt.Errorf("mu has invalid dimension")
	} else if len(sigma) != n*n {
		return nil, fmt.Errorf("sigma has invalid dimension")
	} else {
		Mu := NewVector(BareRealType, mu)
		Si := NewMatrix(BareRealType, n, n, sigma)
		if dist, err := vectorDistribution.NewNormalDistribution(Mu, Si); err != nil {
			return nil, err
		} else {
			r := NormalEstimator{}
			r.NormalDistribution = dist
			r.n = n
			r.SigmaMin = sigmaMin
			return &r, nil
		}
	}
}

/* -------------------------------------------------------------------------- */

func (obj *NormalEstimator) Clone() *NormalEstimator {
	r := NormalEstimator{}
	r.NormalDistribution = obj.NormalDistribution.Clone()
	r.n = obj.n
	r.SigmaMin = obj.SigmaMin
	r.x = obj.x
	return &r
}

func (obj *NormalEstimator) CloneVectorEstimator() VectorEstimator {
	return obj.Clone()
}

func (obj *NormalEstimator) CloneVectorBatchEstimator() VectorBatchEstimator {
	return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *NormalEstimator) Initialize(p ThreadPool) error {
	obj.sum_g = make([]float64, p.NumberOfThreads())
	obj.sum_m = make([][]float64, p.NumberOfThreads())
	obj.sum_s = make([][][]float64, p.NumberOfThreads())
	for i := 0; i < p.NumberOfThreads(); i++ {
		obj.sum_g[i] = 0.0
		obj.sum_m[i] = make([]float64, obj.n)
		obj.sum_s[i] = make([][]float64, obj.n)
		for j := 0; j < obj.n; j++ {
			obj.sum_s[i][j] = make([]float64, obj.n)
		}
	}
	obj.gamma_max = 0.0
	return nil
}

func (obj *NormalEstimator) NewObservation(x ConstVector, gamma ConstScalar, p ThreadPool) error {
	if x.Dim() != obj.n {
		return fmt.Errorf("x has invalid dimension (expected dimension `%d' but data has dimension `%d')", obj.n, x.Dim())
	}
	id := p.GetThreadId()
	if gamma == nil {
		obj.sum_g[id] += 1.0
		for i := 0; i < obj.n; i++ {
			xi := x.ConstAt(i).GetValue()
			obj.sum_m[id][i] += xi
			for j := 0; j < obj.n; j++ {
				xj := x.ConstAt(j).GetValue()
				obj.sum_s[id][i][j] += xi * xj
			}
		}
	} else {
		g := math.Exp(gamma.GetValue() - obj.gamma_max)
		obj.sum_g[id] += g
		for i := 0; i < obj.n; i++ {
			xi := x.ConstAt(i).GetValue()
			obj.sum_m[id][i] += g * xi
			for j := 0; j < obj.n; j++ {
				xj := x.ConstAt(j).GetValue()
				obj.sum_s[id][i][j] += g * xi * xj
			}
		}
	}
	return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *NormalEstimator) estimateParameters() (Vector, Matrix, int) {
	sum_g := obj.sum_g[0]
	sum_m := obj.sum_m[0]
	sum_s := obj.sum_s[0]
	for k := 1; k < len(obj.sum_m); k++ {
		sum_g += obj.sum_g[k]
		for i := 0; i < obj.n; i++ {
			sum_m[i] += obj.sum_m[k][i]
			for j := 0; j < obj.n; j++ {
				sum_s[i][j] += obj.sum_s[k][i][j]
			}
		}
	}
	mu := NullVector(BareRealType, obj.n)
	si := NullMatrix(BareRealType, obj.n, obj.n)
	for i := 0; i < obj.n; i++ {
		mu.At(i).SetValue(sum_m[i] / sum_g)
		for j := 0; j < obj.n; j++ {
			si.At(i, j).SetValue(sum_s[i][j]/sum_g - sum_m[i]/sum_g*sum_m[j]/sum_g)
		}
		if s := si.At(i, i).GetValue(); math.IsNaN(s) || s < obj.SigmaMin {
			si.At(i, i).SetValue(obj.SigmaMin)
		}
	}
	obj.sum_g = nil
	obj.sum_m = nil
	obj.sum_s = nil
	return mu, si, int(math.Round(sum_g))
}

func (obj *NormalEstimator) updateEstimate() error {
	mu, si, _ := obj.estimateParameters()
	if t, err := vectorDistribution.NewNormalDistribution(mu, si); err != nil {
		return err
	} else {
		*obj.NormalDistribution = *t
	}
	return nil
}

func (obj *NormalEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	g := p.NewJobGroup()
	x := obj.x

	// initialize estimator
	obj.Initialize(p)

	// rescale gamma
	//////////////////////////////////////////////////////////////////////////////
	if gamma != nil {
		obj.gamma_max = math.Inf(-1)
		for i := 0; i < gamma.Dim(); i++ {
			if g := gamma.ConstAt(i).GetValue(); obj.gamma_max < g {
				obj.gamma_max = g
			}
		}
	}
	// compute sigma
	//////////////////////////////////////////////////////////////////////////////
	if gamma == nil {
		if err := p.AddRangeJob(0, len(x), g, func(i int, p ThreadPool, erf func() error) error {
			obj.NewObservation(x[i], nil, p)
			return nil
		}); err != nil {
			return err
		}
	} else {
		if err := p.AddRangeJob(0, len(x), g, func(i int, p ThreadPool, erf func() error) error {
			obj.NewObservation(x[i], gamma.ConstAt(i), p)
			return nil
		}); err != nil {
			return err
		}
	}
	if err := p.Wait(g); err != nil {
		return err
	}
	// update estimate
	if err := obj.updateEstimate(); err != nil {
		return err
	}
	return nil
}

func (obj *NormalEstimator) EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *NormalEstimator) GetEstimate() (VectorPdf, error) {
	if obj.sum_m != nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.NormalDistribution, nil
}
