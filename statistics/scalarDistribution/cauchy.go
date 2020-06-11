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

type CauchyDistribution struct {
	Mu    Scalar
	Sigma Scalar
	z     Scalar
	s2    Scalar
}

/* -------------------------------------------------------------------------- */

func NewCauchyDistribution(mu, sigma Scalar) (*CauchyDistribution, error) {
	if sigma.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	t := mu.Type()
	t1 := NewScalar(t, 0.0)
	t2 := NewScalar(t, 0.0)
	r := CauchyDistribution{}
	r.Mu = mu.CloneScalar()
	r.Sigma = sigma.CloneScalar()
	r.z = t1.Log(t1.Div(sigma, ConstReal(math.Pi)))
	r.s2 = t2.Mul(sigma, sigma)
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *CauchyDistribution) Clone() *CauchyDistribution {
	r, _ := NewCauchyDistribution(obj.Mu, obj.Sigma)
	return r
}

func (obj *CauchyDistribution) CloneScalarPdf() ScalarPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *CauchyDistribution) ScalarType() ScalarType {
	return obj.Mu.Type()
}

func (obj *CauchyDistribution) LogPdf(r Scalar, x ConstScalar) error {
	r.Sub(x, obj.Mu)
	r.Mul(r, r)
	r.Add(r, obj.s2)
	r.Log(r)
	// sum up partial results
	r.Sub(obj.z, r)
	return nil
}

func (obj *CauchyDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := obj.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *CauchyDistribution) GetParameters() Vector {
	p := NullVector(obj.ScalarType(), 2)
	p.At(0).Set(obj.Mu)
	p.At(1).Set(obj.Sigma)
	return p
}

func (obj *CauchyDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewCauchyDistribution(parameters.At(0), parameters.At(1)); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *CauchyDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		mu := NewScalar(t, parameters[0])
		sigma := NewScalar(t, parameters[1])

		if tmp, err := NewCauchyDistribution(mu, sigma); err != nil {
			return err
		} else {
			*obj = *tmp
		}
		return nil
	}
}

func (obj *CauchyDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:cauchy distribution", obj.GetParameters())
}
