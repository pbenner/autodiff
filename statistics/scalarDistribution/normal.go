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

package scalarDistribution

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type NormalDistribution struct {
	Mu    Scalar
	Sigma Scalar
}

/* -------------------------------------------------------------------------- */

func NewNormalDistribution(mu, sigma Scalar) (*NormalDistribution, error) {
	if sigma.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	dist := NormalDistribution{}
	dist.Mu = mu.CloneScalar()
	dist.Sigma = sigma.CloneScalar()
	return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) Clone() *NormalDistribution {
	r, _ := NewNormalDistribution(obj.Mu.CloneScalar(), obj.Sigma.CloneScalar())
	return r
}

func (obj *NormalDistribution) CloneScalarPdf() ScalarPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) ScalarType() ScalarType {
	return obj.Mu.Type()
}

func (obj *NormalDistribution) LogPdf(r Scalar, x ConstScalar) error {
	t := obj.ScalarType()

	// z = -1/2 log(2 pi)
	z := NewScalar(t, -0.5*math.Log(2*math.Pi))

	// t1 = -log(sigma)
	t1 := obj.Sigma.CloneScalar()
	t1.Log(t1)
	t1.Neg(t1)
	// t1 = -1/2 log(2 pi) -log(sigma)
	t1.Add(t1, z)

	// t2 = (log(x) - mu)^2/(2 sigma^2)
	t2 := obj.Mu.CloneScalar()
	t2.Sub(x, t2)
	t2.Mul(t2, t2)
	t2.Div(t2, NewBareReal(2.0))
	t2.Div(t2, obj.Sigma)
	t2.Div(t2, obj.Sigma)

	r.Sub(t1, t2)

	return nil
}

func (dist *NormalDistribution) LogCdf(r Scalar, x ConstScalar) error {
	t := dist.Sigma.CloneScalar()
	t.Mul(t, ConstReal(math.Sqrt(2.0)))

	r.Sub(x, dist.Mu)
	r.Div(r, t)
	r.Neg(r)
	r.LogErfc(r)
	r.Sub(r, ConstReal(math.Log(2.0)))

	// if computation of derivatives failed, return an approximation
	if r.GetOrder() >= 1 {
		for i := 0; i < r.GetN(); i++ {
			if math.IsNaN(r.GetDerivative(i)) && x.GetValue() < 0.0 {
				r.SetDerivative(i, -x.GetValue()*x.GetDerivative(i))
			}
		}
	}
	return nil
}

func (dist *NormalDistribution) Cdf(r Scalar, x ConstScalar) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) GetParameters() Vector {
	p := NullVector(obj.ScalarType(), 2)
	p.At(0).Set(obj.Mu)
	p.At(1).Set(obj.Sigma)
	return p
}

func (obj *NormalDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewNormalDistribution(parameters.At(0), parameters.At(1)); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		mu := NewScalar(t, parameters[0])
		sigma := NewScalar(t, parameters[1])

		if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
			return err
		} else {
			*obj = *tmp
		}
		return nil
	}
}

func (obj *NormalDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:normal distribution", obj.GetParameters())
}
