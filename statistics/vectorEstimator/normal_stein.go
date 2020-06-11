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

package vectorEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/eigensystem"
import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type NormalSteinEstimator struct {
	NormalEstimator
}

/* -------------------------------------------------------------------------- */

func NewNormalSteinEstimator(mu, sigma []float64, sigmaMin float64) (*NormalSteinEstimator, error) {
	if r, err := NewNormalEstimator(mu, sigma, sigmaMin); err != nil {
		return nil, err
	} else {
		return &NormalSteinEstimator{*r}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj NormalSteinEstimator) steinEigen(l ConstVector, n int) Vector {
	r := NullVector(l.ElementType(), l.Dim())
	t := NullScalar(l.ElementType())
	p := l.Dim()
	// k = min(p, n)
	k := p
	if k > n {
		k = n
	}
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			if i != j {
				t.Sub(l.ConstAt(i), l.ConstAt(j))
				t.Div(ConstReal(1.0), t)
				r.At(i).Add(r.ConstAt(i), t)
			}
		}
		t.Mul(ConstReal(2.0), l.ConstAt(i))
		r.At(i).Mul(r.ConstAt(i), t)
		r.At(i).Add(r.ConstAt(i), ConstReal(1.0))
		r.At(i).Add(r.ConstAt(i), ConstReal(math.Abs(float64(n)-float64(p))))
		t.Mul(ConstReal(float64(n)), l.ConstAt(i))
		r.At(i).Div(t, r.ConstAt(i))
	}
	r.Sort(true)
	return r
}

/* -------------------------------------------------------------------------- */

func (obj *NormalSteinEstimator) updateEstimate() error {
	// compute empirical mean and covariance
	mu, si, n := obj.estimateParameters()
	// compute eigensystem of the covariance matrix
	l, H, err := eigensystem.Run(si, eigensystem.Symmetric{true})
	if err != nil {
		return err
	}
	// compute Stein's eigenvalues
	l = obj.steinEigen(l, n)
	// l * H^t
	for i := 0; i < l.Dim(); i++ {
		for j := 0; j < l.Dim(); j++ {
			si.At(i, j).Mul(H.ConstAt(j, i), l.ConstAt(i))
		}
	}
	si.MdotM(H, si)
	// create new estimate
	if t, err := vectorDistribution.NewNormalDistribution(mu, si); err != nil {
		return err
	} else {
		*obj.NormalDistribution = *t
	}
	return nil
}

func (obj *NormalSteinEstimator) GetEstimate() (VectorPdf, error) {
	if obj.sum_m != nil {
		if err := obj.updateEstimate(); err != nil {
			return nil, err
		}
	}
	return obj.NormalDistribution, nil
}
