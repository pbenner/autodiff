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

type LaplaceDistribution struct {
	Mu    Scalar
	Sigma Scalar
	c1    Scalar
	c2    Scalar
}

/* -------------------------------------------------------------------------- */

func NewLaplaceDistribution(mu, sigma Scalar) (*LaplaceDistribution, error) {

	result := LaplaceDistribution{}
	result.Mu = mu.CloneScalar()
	result.Sigma = sigma.CloneScalar()
	result.c1 = NewScalar(mu.Type(), 1.0)
	result.c2 = NewScalar(mu.Type(), 2.0)

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *LaplaceDistribution) Clone() *LaplaceDistribution {
	return &LaplaceDistribution{
		Mu:    dist.Mu.CloneScalar(),
		Sigma: dist.Sigma.CloneScalar(),
		c1:    dist.c1.CloneScalar(),
		c2:    dist.c2.CloneScalar()}
}

func (obj *LaplaceDistribution) CloneScalarPdf() ScalarPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *LaplaceDistribution) ScalarType() ScalarType {
	return dist.Mu.Type()
}

func (dist *LaplaceDistribution) LogPdf(r Scalar, x ConstScalar) error {

	r.Sub(x, dist.Mu)
	r.Abs(r)
	r.Div(r, dist.Sigma)
	r.Neg(r)
	r.Exp(r)
	r.Div(r, dist.Sigma)
	r.Div(r, dist.c2)

	return nil
}

func (dist *LaplaceDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *LaplaceDistribution) LogCdf(r Scalar, x Vector) error {

	r.Sub(x.At(0), dist.Mu)
	r.Abs(r)
	r.Div(r, dist.Sigma)
	r.Neg(r)
	r.Exp(r)
	r.Div(r, dist.c2)

	if x.At(0).Greater(dist.Mu) {
		r.Neg(r)
		r.Add(r, dist.c1)
	}
	return nil
}

func (dist *LaplaceDistribution) Cdf(r Scalar, x Vector) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LaplaceDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 2)
	p.At(0).Set(dist.Mu)
	p.At(1).Set(dist.Sigma)
	return p
}

func (dist *LaplaceDistribution) SetParameters(parameters Vector) error {
	mu := parameters.At(0)
	sigma := parameters.At(1)
	if tmp, err := NewLaplaceDistribution(mu, sigma); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *LaplaceDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		mu := NewScalar(t, parameters[0])
		sigma := NewScalar(t, parameters[1])

		if tmp, err := NewLaplaceDistribution(mu, sigma); err != nil {
			return err
		} else {
			*obj = *tmp
		}
		return nil
	}
}

func (obj *LaplaceDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:laplace distribution", obj.GetParameters())
}
