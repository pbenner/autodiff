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

type VectorId struct {
	Distributions []VectorPdf
	n             int
	t             Scalar
}

/* -------------------------------------------------------------------------- */

func NewVectorId(distributions ...VectorPdf) (*VectorId, error) {
	if len(distributions) == 0 {
		return &VectorId{}, nil
	}
	d := make([]VectorPdf, len(distributions))
	t := NewScalar(distributions[0].ScalarType(), 0.0)
	n := 0
	for i := 0; i < len(distributions); i++ {
		n += distributions[i].Dim()
		d[i] = distributions[i].CloneVectorPdf()
	}
	return &VectorId{d, n, t}, nil
}

/* -------------------------------------------------------------------------- */

func (obj *VectorId) Clone() *VectorId {
	r, _ := NewVectorId(obj.Distributions...)
	return r
}

func (obj *VectorId) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *VectorId) Dim() int {
	return obj.n
}

func (obj *VectorId) ScalarType() ScalarType {
	return obj.Distributions[0].ScalarType()
}

func (obj *VectorId) LogPdf(r Scalar, x ConstVector) error {
	if x.Dim() != obj.Dim() {
		return fmt.Errorf("LogPdf(): dimensions do not match (input has dimension `%d' whereas this distribution is of dimension `%d'", x.Dim(), obj.Dim())
	}
	r.Reset()
	for i, j := 0, 0; i < len(obj.Distributions); i++ {
		m := obj.Distributions[i].Dim()
		if err := obj.Distributions[i].LogPdf(obj.t, x.ConstSlice(j, j+m)); err != nil {
			return err
		}
		r.Add(r, obj.t)
		j += m
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *VectorId) GetParameters() Vector {
	if len(obj.Distributions) == 0 {
		return nil
	}
	p := obj.Distributions[0].GetParameters()
	for i := 1; i < len(obj.Distributions); i++ {
		p = p.AppendVector(obj.Distributions[i].GetParameters())
	}
	return p
}

func (obj *VectorId) SetParameters(parameters Vector) error {
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

func (obj *VectorId) ImportConfig(config ConfigDistribution, t ScalarType) error {
	distributions := []VectorPdf{}

	for i := 0; i < len(config.Distributions); i++ {
		if obj, err := ImportVectorPdfConfig(config.Distributions[i], t); err != nil {
			return err
		} else {
			distributions = append(distributions, obj)
		}
	}
	if tmp, err := NewVectorId(distributions...); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *VectorId) ExportConfig() (config ConfigDistribution) {
	distributions := make([]ConfigDistribution, len(obj.Distributions))

	for i := 0; i < len(obj.Distributions); i++ {
		distributions[i] = obj.Distributions[i].ExportConfig()
	}

	return NewConfigDistribution("vector:vector id", nil, distributions...)
}
