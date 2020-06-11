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

type GevDistribution struct {
	Mu    Scalar
	Sigma Scalar
	Xi    Scalar
	c1    Scalar
	cx    Scalar
	cy    Scalar
	t     Scalar
}

/* -------------------------------------------------------------------------- */

func NewGevDistribution(mu, sigma, xi Scalar) (*GevDistribution, error) {
	if sigma.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid value for parameter sigma: %f", sigma.GetValue())
	}
	// some constants
	c1 := NewBareReal(1.0)
	cx := NewScalar(xi.Type(), 1.0)
	cx.Div(cx, xi)
	cx.Neg(cx)
	cy := NewScalar(xi.Type(), 1.0)
	cy.Div(cy, xi)
	cy.Add(cy, c1)

	result := GevDistribution{
		Mu:    mu.CloneScalar(),
		Sigma: sigma.CloneScalar(),
		Xi:    xi.CloneScalar(),
		c1:    c1,
		cx:    cx,
		cy:    cy,
		t:     NewScalar(xi.Type(), 0.0)}

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) Clone() *GevDistribution {
	return &GevDistribution{
		Mu:    dist.Mu.CloneScalar(),
		Sigma: dist.Sigma.CloneScalar(),
		Xi:    dist.Xi.CloneScalar(),
		c1:    dist.c1.CloneScalar(),
		cx:    dist.cx.CloneScalar(),
		cy:    dist.cy.CloneScalar(),
		t:     dist.t.CloneScalar()}
}

func (dist *GevDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) ScalarType() ScalarType {
	return dist.Mu.Type()
}

func (dist *GevDistribution) LogPdf(r Scalar, x ConstScalar) error {

	if dist.Xi.GetValue()*(x.GetValue()-dist.Mu.GetValue())/dist.Sigma.GetValue() <= -1 {
		r.SetValue(math.Inf(-1))
		return nil
	}
	t := dist.t
	t.Sub(x, dist.Mu)
	t.Div(t, dist.Sigma)

	if dist.Xi.GetValue() == 0.0 {
		t.Neg(t)
		// r = - (x-mu)/sigma
		r.Set(t)
		t.Exp(t)
		// r = - (x-mu)/sigma - exp{-(x-mu)/sigma}
		r.Sub(r, t)

	} else {
		t.Mul(t, dist.Xi)
		t.Add(t, dist.c1)
		r.Pow(t, dist.cx)
		// r = - (1 + xi(x-mu)/sigma)^(-1/xi)
		r.Neg(r)

		t.Log(t)
		t.Mul(t, dist.cy)
		// r = - (1+1/xi) log(1 + xi(x-mu)/sigma) - (1 + xi(x-mu)/sigma)^(-1/xi)
		r.Sub(r, t)
	}
	t.Log(dist.Sigma)
	r.Sub(r, t)

	return nil
}

func (dist *GevDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *GevDistribution) LogCdf(r Scalar, x ConstScalar) error {
	if dist.Xi.GetValue()*(x.GetValue()-dist.Mu.GetValue())/dist.Sigma.GetValue() <= -1 {
		r.SetValue(math.Inf(-1))
		return nil
	}
	r.Set(x)
	r.Sub(r, dist.Mu)
	r.Div(r, dist.Sigma)

	if dist.Xi.GetValue() == 0.0 {
		r.Neg(r)
		r.Exp(r)
		r.Neg(r)
	} else {
		r.Mul(r, dist.Xi)
		r.Add(r, dist.c1)
		r.Pow(r, dist.cx)
		r.Neg(r)
	}

	return nil
}

func (dist *GevDistribution) Cdf(r Scalar, x ConstScalar) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 3)
	p.At(0).Set(dist.Mu)
	p.At(1).Set(dist.Sigma)
	p.At(2).Set(dist.Xi)
	return p
}

func (dist *GevDistribution) SetParameters(parameters Vector) error {
	mu := parameters.At(0)
	sigma := parameters.At(1)
	xi := parameters.At(2)
	if tmp, err := NewGevDistribution(mu, sigma, xi); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		mu := NewScalar(t, parameters[0])
		sigma := NewScalar(t, parameters[1])
		xi := NewScalar(t, parameters[2])

		if tmp, err := NewGevDistribution(mu, sigma, xi); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *GevDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:gev distribution", dist.GetParameters())
}
