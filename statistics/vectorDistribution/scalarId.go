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

package vectorDistribution

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ScalarId struct {
	Distributions []ScalarPdf
	t             Scalar
}

/* -------------------------------------------------------------------------- */

func NewScalarId(distributions ...ScalarPdf) (*ScalarId, error) {
	if len(distributions) == 0 {
		return &ScalarId{}, nil
	}
	d := make([]ScalarPdf, len(distributions))
	t := NewScalar(distributions[0].ScalarType(), 0.0)
	for i := 0; i < len(distributions); i++ {
		d[i] = distributions[i].CloneScalarPdf()
	}
	return &ScalarId{d, t}, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarId) Clone() *ScalarId {
	r, _ := NewScalarId(obj.Distributions...)
	return r
}

func (obj *ScalarId) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarId) Dim() int {
	return len(obj.Distributions)
}

func (obj *ScalarId) ScalarType() ScalarType {
	return obj.Distributions[0].ScalarType()
}

func (obj *ScalarId) LogPdf(r Scalar, x ConstVector) error {
	if x.Dim() != obj.Dim() {
		return fmt.Errorf("LogPdf(): dimensions do not match (input has dimension `%d' whereas this distribution is of dimension `%d'", x.Dim(), obj.Dim())
	}
	r.Reset()
	for i := 0; i < len(obj.Distributions); i++ {
		if err := obj.Distributions[i].LogPdf(obj.t, x.ConstAt(i)); err != nil {
			return err
		}
		r.Add(r, obj.t)
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarId) GetParameters() Vector {
	if len(obj.Distributions) == 0 {
		return nil
	}
	p := obj.Distributions[0].GetParameters()
	for i := 1; i < len(obj.Distributions); i++ {
		p = p.AppendVector(obj.Distributions[i].GetParameters())
	}
	return p
}

func (obj *ScalarId) SetParameters(parameters Vector) error {
	for i := 0; i < len(obj.Distributions); i++ {
		n := obj.Distributions[i].GetParameters().Dim()
		if parameters.Dim() < n {
			return fmt.Errorf("invalid set of parameters")
		}
		if err := obj.Distributions[i].SetParameters(parameters.Slice(0, n)); err != nil {
			return err
		}
		parameters = parameters.Slice(n, parameters.Dim())
	}
	if parameters.Dim() != 0 {
		return fmt.Errorf("invalid set of parameters")
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarId) ImportConfig(config ConfigDistribution, t ScalarType) error {
	distributions := []ScalarPdf{}

	for i := 0; i < len(config.Distributions); i++ {
		if obj, err := ImportScalarPdfConfig(config.Distributions[i], t); err != nil {
			return err
		} else {
			distributions = append(distributions, obj)
		}
	}
	if tmp, err := NewScalarId(distributions...); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *ScalarId) ExportConfig() (config ConfigDistribution) {
	distributions := make([]ConfigDistribution, len(obj.Distributions))

	for i := 0; i < len(obj.Distributions); i++ {
		distributions[i] = obj.Distributions[i].ExportConfig()
	}

	return NewConfigDistribution("vector:scalar id", nil, distributions...)
}
