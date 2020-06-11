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

// Gamma(r+k)/Gamma(k+1)/Gamma(r) p^k (1-p)^r

type NegativeBinomialDistribution struct {
	R  Scalar
	P  Scalar
	p  Scalar // q = log(p)
	z  Scalar
	c1 Scalar
	t1 Scalar
	t2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewNegativeBinomialDistribution(r, p Scalar) (*NegativeBinomialDistribution, error) {
	if r.GetValue() <= 0.0 || p.GetValue() < 0.0 || p.GetValue() > 1.0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	t := r.Type()

	// (1-p)^r
	t1 := p.CloneScalar()
	t1.Sub(NewBareReal(1.0), t1)
	t1.Log(t1)
	t1.Mul(t1, r)

	// Gamma(r)
	t2 := r.CloneScalar()
	t2.Lgamma(t2)

	// p^r / Gamma(r)
	t1.Sub(t1, t2)

	dist := NegativeBinomialDistribution{}
	dist.R = r.CloneScalar()
	dist.P = p.CloneScalar()
	dist.p = NewScalar(t, 0.0)
	dist.p.Log(p)
	dist.z = t1
	dist.c1 = NewScalar(t, 1.0)
	dist.t1 = NewScalar(t, 0.0)
	dist.t2 = NewScalar(t, 0.0)
	return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) Clone() *NegativeBinomialDistribution {
	r, _ := NewNegativeBinomialDistribution(dist.R, dist.P)
	return r
}

func (dist *NegativeBinomialDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) ScalarType() ScalarType {
	return dist.R.Type()
}

func (dist *NegativeBinomialDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); v < 0.0 || math.Floor(v) != v {
		r.SetValue(math.Inf(-1))
		return nil
	}
	t1 := dist.t1
	t2 := dist.t2

	// Gamma(r + k)
	t1.Add(dist.R, x)
	t1.Lgamma(t1)

	// Gamma(k + 1)
	t2.Add(x, dist.c1)
	t2.Lgamma(t2)

	r.Mul(x, dist.p)
	r.Add(r, t1)
	r.Sub(r, t2)
	r.Add(r, dist.z)

	return nil
}

func (dist *NegativeBinomialDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 2)
	p.At(0).Set(dist.R)
	p.At(1).Set(dist.P)
	return p
}

func (dist *NegativeBinomialDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewNegativeBinomialDistribution(parameters.At(0), parameters.At(1)); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		r := NewScalar(t, parameters[0])
		p := NewScalar(t, parameters[1])

		if tmp, err := NewNegativeBinomialDistribution(r, p); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *NegativeBinomialDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:negative binomial distribution", dist.GetParameters())
}
