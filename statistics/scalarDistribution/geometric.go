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

type GeometricDistribution struct {
	p  Scalar
	p1 Scalar // log p
	p2 Scalar // log (1-p)
}

/* -------------------------------------------------------------------------- */

func NewGeometricDistribution(p Scalar) (*GeometricDistribution, error) {
	if p.GetValue() <= 0.0 || p.GetValue() > 1.0 {
		return nil, fmt.Errorf("invalid value for parameter p: %f", p.GetValue())
	}

	t := p.Type()
	p1 := NewScalar(t, 0.0)
	p2 := NewScalar(t, 0.0)

	p1.Log(p)
	p2.Sub(ConstReal(1.0), p)
	p2.Log(p2)

	result := GeometricDistribution{
		p:  p.CloneScalar(),
		p1: p1,
		p2: p2}

	return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeometricDistribution) Clone() *GeometricDistribution {
	return &GeometricDistribution{
		p:  dist.p.CloneScalar(),
		p1: dist.p1.CloneScalar(),
		p2: dist.p2.CloneScalar()}
}

func (dist *GeometricDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GeometricDistribution) ScalarType() ScalarType {
	return dist.p.Type()
}

func (dist *GeometricDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); math.Floor(v) != v {
		return fmt.Errorf("value `%f' is not an integer", v)
	}

	r.Mul(x, dist.p2)
	r.Add(r, dist.p1)

	return nil
}

func (dist *GeometricDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist GeometricDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 1)
	p.At(0).Set(dist.p)
	return p
}

func (dist *GeometricDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewGeometricDistribution(parameters.At(0)); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeometricDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		p := NewScalar(t, parameters[0])

		if tmp, err := NewGeometricDistribution(p); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *GeometricDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:geometric distribution", dist.GetParameters())
}
