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

type PoissonDistribution struct {
	Lambda Scalar
	t      Scalar
}

/* -------------------------------------------------------------------------- */

func NewPoissonDistribution(lambda Scalar) (*PoissonDistribution, error) {

	if lambda.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid parameter")
	}

	result := PoissonDistribution{}
	result.Lambda = lambda.CloneScalar()
	result.t = lambda.CloneScalar()

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *PoissonDistribution) Clone() *PoissonDistribution {
	return &PoissonDistribution{
		Lambda: dist.Lambda.CloneScalar(),
		t:      dist.t.CloneScalar()}
}

func (dist *PoissonDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *PoissonDistribution) ScalarType() ScalarType {
	return dist.Lambda.Type()
}

func (dist *PoissonDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); math.Floor(v) != v {
		return fmt.Errorf("value `%f' is not an integer", v)
	}
	if v := x.GetValue(); v < 0.0 {
		r.SetValue(math.Inf(-1))
		return nil
	}

	t := dist.t
	// k! = Gamma(k+1)
	t.Add(x, ConstReal(1.0))
	t.Lgamma(t)

	// lambda^k
	r.Log(dist.Lambda)
	r.Mul(r, x)

	// lambda^k/Gamma(k+1)
	r.Sub(r, t)
	// lambda^k Exp(-lambda)/Gamma(k+1)
	r.Sub(r, dist.Lambda)

	return nil
}

func (dist *PoissonDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *PoissonDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 1)
	p.At(0).Set(dist.Lambda)
	return p
}

func (dist *PoissonDistribution) SetParameters(parameters Vector) error {
	lambda := parameters.At(0)
	if tmp, err := NewPoissonDistribution(lambda); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *PoissonDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		lambda := NewScalar(t, parameters[0])

		if tmp, err := NewPoissonDistribution(lambda); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *PoissonDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:poisson distribution", dist.GetParameters())
}
