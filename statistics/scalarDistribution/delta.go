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

type DeltaDistribution struct {
	X Scalar
}

/* -------------------------------------------------------------------------- */

func NewDeltaDistribution(x Scalar) (*DeltaDistribution, error) {
	return &DeltaDistribution{x.CloneScalar()}, nil
}

/* -------------------------------------------------------------------------- */

func (dist *DeltaDistribution) Clone() *DeltaDistribution {
	return &DeltaDistribution{dist.X.CloneScalar()}
}

func (dist *DeltaDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *DeltaDistribution) ScalarType() ScalarType {
	return dist.X.Type()
}

func (dist *DeltaDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if x.GetValue() == dist.X.GetValue() {
		r.SetValue(0.0)
	} else {
		r.SetValue(math.Inf(-1))
	}
	return nil
}

func (dist *DeltaDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *DeltaDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 1)
	p.At(0).Set(dist.X)
	return p
}

func (dist *DeltaDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewDeltaDistribution(parameters.At(0)); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *DeltaDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		x := NewScalar(t, parameters[0])

		if tmp, err := NewDeltaDistribution(x); err != nil {
			return err
		} else {
			*obj = *tmp
		}
		return nil
	}
}

func (obj *DeltaDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:delta distribution", obj.GetParameters())
}
