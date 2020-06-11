/* Copyright (C) 2016 Philipp Benner
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

package scalarDistribution

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type ChiSquaredDistribution struct {
	K Scalar
	C Scalar // c = 2
	L Scalar // l = k/2
	E Scalar // k/2 - 1
	Z Scalar // k/2 log 2 + log Gamma(k/2)
}

/* -------------------------------------------------------------------------- */

func NewChiSquaredDistribution(t ScalarType, k_ float64) (*ChiSquaredDistribution, error) {
	// we cannot differentiate with respect to k, so use bare reals
	k := NewScalar(t, k_)
	c1 := NewScalar(t, 1.0)
	c2 := NewScalar(t, 2.0)
	l := NewScalar(t, 0.0)
	l.Div(k, c2)
	e := NewScalar(t, 0.0)
	e.Sub(l, c1)
	z := NewScalar(t, 0.0)
	t1 := NewScalar(t, 0.0)
	z.Add(z.Mul(l, z.Log(c2)), t1.Lgamma(l))
	return &ChiSquaredDistribution{K: k, C: c2, L: l, E: e, Z: z}, nil
}

/* -------------------------------------------------------------------------- */

func (dist *ChiSquaredDistribution) Clone() *ChiSquaredDistribution {
	r, _ := NewChiSquaredDistribution(dist.ScalarType(), dist.K.GetValue())
	return r
}

func (dist *ChiSquaredDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *ChiSquaredDistribution) ScalarType() ScalarType {
	return dist.K.Type()
}

func (dist *ChiSquaredDistribution) LogPdf(r Scalar, x ConstScalar) error {
	t := NewScalar(dist.ScalarType(), 0.0)
	r.Log(x)
	r.Mul(r, dist.E)
	t.Div(x, dist.C)
	r.Sub(r, t)
	r.Sub(r, dist.Z)
	return nil
}

func (dist *ChiSquaredDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *ChiSquaredDistribution) LogCdf(r Scalar, x ConstScalar) error {
	if err := dist.Cdf(r, x); err != nil {
		return err
	}
	r.Log(r)
	return nil
}

func (dist *ChiSquaredDistribution) Cdf(r Scalar, x ConstScalar) error {
	r.Div(x, dist.C)
	r.GammaP(dist.L.GetValue(), r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *ChiSquaredDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 1)
	p.At(0).Set(dist.K)
	return p
}

func (dist *ChiSquaredDistribution) SetParameters(parameters Vector) error {
	k := parameters.At(0)
	if tmp, err := NewChiSquaredDistribution(dist.ScalarType(), k.GetValue()); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *ChiSquaredDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		if tmp, err := NewChiSquaredDistribution(t, parameters[0]); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *ChiSquaredDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:chi-squared distribution", dist.GetParameters())
}
