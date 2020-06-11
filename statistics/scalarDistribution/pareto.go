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

type ParetoDistribution struct {
	Lambda  Scalar // scale
	Kappa   Scalar // shape
	Kappa1p Scalar
	z       Scalar
}

/* -------------------------------------------------------------------------- */

func NewParetoDistribution(lambda, kappa Scalar) (*ParetoDistribution, error) {
	if lambda.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid value for parameter lambda: %f", lambda.GetValue())
	}
	if kappa.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid value for parameter kappa: %f", kappa.GetValue())
	}

	t := lambda.Type()
	t1 := NewScalar(t, 0.0)
	t2 := NewScalar(t, 0.0)

	kappa1p := kappa.CloneScalar()
	kappa1p.Add(kappa1p, NewBareReal(1.0))

	z := t1.Add(t1.Log(kappa), t2.Mul(kappa, t2.Log(lambda)))

	result := &ParetoDistribution{
		Lambda:  lambda.CloneScalar(),
		Kappa:   kappa.CloneScalar(),
		Kappa1p: kappa1p.CloneScalar(),
		z:       z}

	return result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) Clone() *ParetoDistribution {
	return &ParetoDistribution{
		Lambda:  dist.Lambda.CloneScalar(),
		Kappa:   dist.Kappa.CloneScalar(),
		Kappa1p: dist.Kappa1p.CloneScalar(),
		z:       dist.z.CloneScalar()}
}

func (dist *ParetoDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) ScalarType() ScalarType {
	return dist.Lambda.Type()
}

func (dist *ParetoDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if x.GetValue() < 0 {
		r.SetValue(math.Inf(-1))
		return nil
	}

	r.Log(x)
	r.Mul(r, dist.Kappa1p)
	r.Neg(r)
	r.Add(r, dist.z)

	return nil
}

func (dist *ParetoDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *ParetoDistribution) LogCdf(r Scalar, x ConstScalar) error {
	if x.GetValue() < 0 {
		r.SetValue(math.Inf(-1))
		return nil
	}

	r.Div(dist.Lambda, x)
	r.Pow(r, dist.Kappa)
	r.Neg(r)
	r.Log1p(r)

	return nil
}

func (dist *ParetoDistribution) Cdf(r Scalar, x ConstScalar) error {
	if err := dist.LogCdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist ParetoDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 2)
	p.At(0).Set(dist.Lambda)
	p.At(1).Set(dist.Kappa)
	return p
}

func (dist *ParetoDistribution) SetParameters(parameters Vector) error {
	lambda := parameters.At(0)
	kappa := parameters.At(1)
	if tmp, err := NewParetoDistribution(lambda, kappa); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		lambda := NewScalar(t, parameters[0])
		kappa := NewScalar(t, parameters[1])

		if tmp, err := NewParetoDistribution(lambda, kappa); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *ParetoDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:pareto distribution", dist.GetParameters())
}
