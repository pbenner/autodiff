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

type ExponentialDistribution struct {
	Lambda    Scalar
	LambdaLog Scalar
	c1        Scalar
}

/* -------------------------------------------------------------------------- */

func NewExponentialDistribution(lambda Scalar) (*ExponentialDistribution, error) {
	if lambda.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid value for parameter lambda: %f", lambda.GetValue())
	}
	// some constants
	l1 := lambda.CloneScalar()
	l2 := lambda.CloneScalar()
	l2.Log(l1)

	result := ExponentialDistribution{
		Lambda:    l1,
		LambdaLog: l2}

	return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) Clone() *ExponentialDistribution {
	return &ExponentialDistribution{
		Lambda:    dist.Lambda.CloneScalar(),
		LambdaLog: dist.LambdaLog.CloneScalar()}
}

func (dist *ExponentialDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) ScalarType() ScalarType {
	return dist.Lambda.Type()
}

func (dist *ExponentialDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if x.GetValue() < 0 {
		r.SetValue(math.Inf(-1))
		return nil
	}

	r.Mul(dist.Lambda, x)
	r.Neg(r)
	r.Add(r, dist.LambdaLog)

	return nil
}

func (dist *ExponentialDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *ExponentialDistribution) LogCdf(r Scalar, x ConstScalar) error {
	if x.GetValue() < 0 {
		r.SetValue(math.Inf(-1))
		return nil
	}

	r.Mul(dist.Lambda, x)
	r.Neg(r)
	r.Exp(r)
	r.Neg(r)
	r.Log1p(r)

	return nil
}

func (dist *ExponentialDistribution) Cdf(r Scalar, x ConstScalar) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist ExponentialDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 1)
	p.At(0).Set(dist.Lambda)
	return p
}

func (dist *ExponentialDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewExponentialDistribution(parameters.At(0)); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		lambda := NewScalar(t, parameters[0])

		if tmp, err := NewExponentialDistribution(lambda); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *ExponentialDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:exponential distribution", dist.GetParameters())
}
