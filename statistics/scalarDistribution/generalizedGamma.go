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
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type GeneralizedGammaDistribution struct {
	A   Scalar
	D   Scalar
	P   Scalar
	dm1 Scalar
	z   Scalar
	t   Scalar
}

/* -------------------------------------------------------------------------- */

func NewGeneralizedGammaDistribution(a, d, p Scalar) (*GeneralizedGammaDistribution, error) {
	if a.GetValue() <= 0.0 || d.GetValue() <= 0.0 || p.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	t := a.Type()
	t1 := NewScalar(t, 0.0)
	dist := GeneralizedGammaDistribution{}
	dist.A = a.CloneScalar()
	dist.D = d.CloneScalar()
	dist.P = p.CloneScalar()
	dist.dm1 = NewScalar(t, 0.0)
	dist.dm1.Sub(d, NewScalar(d.Type(), 1.0))
	dist.z = NewScalar(t, 0.0)
	dist.z.Log(p)
	dist.z.Sub(dist.z, t1.Mul(d, t1.Log(a)))
	dist.z.Sub(dist.z, t1.Lgamma(t1.Div(d, p)))
	dist.t = t1
	return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) Clone() *GeneralizedGammaDistribution {
	r, _ := NewGeneralizedGammaDistribution(dist.A, dist.D, dist.P)
	return r
}

func (dist *GeneralizedGammaDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) ScalarType() ScalarType {
	return dist.A.Type()
}

func (dist *GeneralizedGammaDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); v <= 0.0 || math.IsInf(v, 1) {
		r.SetValue(math.Inf(-1))
		return nil
	}
	t := dist.t
	t.Div(x, dist.A)
	t.Pow(t, dist.P)

	r.Log(x)
	r.Mul(r, dist.dm1)
	r.Sub(r, t)
	r.Add(r, dist.z)
	return nil
}

func (dist *GeneralizedGammaDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 3)
	p.At(0).Set(dist.A)
	p.At(1).Set(dist.D)
	p.At(2).Set(dist.P)
	return p
}

func (dist *GeneralizedGammaDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewGeneralizedGammaDistribution(parameters.At(0), parameters.At(1), parameters.At(2)); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		a := NewScalar(t, parameters[0])
		d := NewScalar(t, parameters[1])
		p := NewScalar(t, parameters[2])

		if tmp, err := NewGeneralizedGammaDistribution(a, d, p); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *GeneralizedGammaDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:generalized gamma distribution", dist.GetParameters())
}
