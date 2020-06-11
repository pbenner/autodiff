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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type GParetoDistribution struct {
	Mu    Scalar
	Sigma Scalar
	Xi    Scalar
	c1    Scalar
	cx1   Scalar
	cx2   Scalar
	cs    Scalar
}

/* -------------------------------------------------------------------------- */

func NewGParetoDistribution(mu, sigma, xi Scalar) (*GParetoDistribution, error) {
	if sigma.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid value for parameter sigma: %f", sigma.GetValue())
	}
	// some constants
	c1 := NewBareReal(1.0)
	// cx1 = -1/xi
	cx1 := NewScalar(xi.Type(), 1.0)
	cx1.Div(cx1, xi)
	cx1.Neg(cx1)
	// cx2 = -1/xi - 1
	cx2 := cx1.CloneScalar()
	cx2.Sub(cx2, c1)
	cs := sigma.CloneScalar()
	cs.Log(sigma)

	result := &GParetoDistribution{
		Mu:    mu.CloneScalar(),
		Sigma: sigma.CloneScalar(),
		Xi:    xi.CloneScalar(),
		c1:    c1,
		cx1:   cx1,
		cx2:   cx2,
		cs:    cs}

	return result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GParetoDistribution) Clone() *GParetoDistribution {
	return &GParetoDistribution{
		Mu:    dist.Mu.CloneScalar(),
		Sigma: dist.Sigma.CloneScalar(),
		Xi:    dist.Xi.CloneScalar(),
		c1:    dist.c1.CloneScalar(),
		cx1:   dist.cx1.CloneScalar(),
		cx2:   dist.cx2.CloneScalar(),
		cs:    dist.cs.CloneScalar()}
}

func (dist *GParetoDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GParetoDistribution) ScalarType() ScalarType {
	return dist.Mu.Type()
}

func (dist *GParetoDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if dist.Xi.GetValue() >= 0 {
		// xi >= 0
		if x.GetValue() < dist.Mu.GetValue() {
			r.SetValue(math.Inf(-1))
			return nil
		}
	} else {
		// xi < 0
		if x.GetValue() < dist.Mu.GetValue() || x.GetValue() > dist.Mu.GetValue()-dist.Sigma.GetValue()/dist.Xi.GetValue() {
			r.SetValue(math.Inf(-1))
			return nil
		}
	}

	r.Sub(r, dist.Mu)
	r.Div(r, dist.Sigma)

	if dist.Xi.GetValue() == 0.0 {
		r.Neg(r)
	} else {
		r.Mul(r, dist.Xi)
		r.Log1p(r)
		r.Mul(r, dist.cx2) // cx2 = -1/xi - 1
		r.Sub(r, dist.cs)  // cs  = log sigma
	}

	return nil
}

func (dist *GParetoDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *GParetoDistribution) LogCdf(r Scalar, x ConstScalar) error {
	if dist.Xi.GetValue() >= 0 {
		// xi >= 0
		if x.GetValue() < dist.Mu.GetValue() {
			r.SetValue(math.Inf(-1))
			return nil
		}
	} else {
		// xi < 0
		if x.GetValue() < dist.Mu.GetValue() || x.GetValue() > dist.Mu.GetValue()-dist.Sigma.GetValue()/dist.Xi.GetValue() {
			r.SetValue(math.Inf(-1))
			return nil
		}
	}
	r.Sub(r, dist.Mu)
	r.Div(r, dist.Sigma)

	if dist.Xi.GetValue() == 0.0 {
		r.Neg(r)
		r.Exp(r)
	} else {
		r.Mul(r, dist.Xi)
		r.Add(r, dist.c1)
		r.Pow(r, dist.cx1)
	}
	r.Neg(r)
	r.Log1p(r)

	return nil
}

func (dist *GParetoDistribution) Cdf(r Scalar, x ConstScalar) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GParetoDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 3)
	p.At(0).Set(dist.Mu)
	p.At(1).Set(dist.Sigma)
	p.At(2).Set(dist.Xi)
	return p
}

func (dist *GParetoDistribution) SetParameters(parameters Vector) error {
	mu := parameters.At(0)
	sigma := parameters.At(1)
	xi := parameters.At(2)
	if tmp, err := NewGParetoDistribution(mu, sigma, xi); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GParetoDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		mu := NewScalar(t, parameters[0])
		sigma := NewScalar(t, parameters[1])
		xi := NewScalar(t, parameters[2])

		if tmp, err := NewGParetoDistribution(mu, sigma, xi); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *GParetoDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:generalized pareto distribution", dist.GetParameters())
}
