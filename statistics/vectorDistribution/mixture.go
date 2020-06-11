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

package vectorDistribution

/* -------------------------------------------------------------------------- */

import "fmt"
import "bytes"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Mixture struct {
	generic.Mixture
	Edist []VectorPdf
}

/* -------------------------------------------------------------------------- */

func NewMixture(weights Vector, edist []VectorPdf) (*Mixture, error) {
	if mixture, err := generic.NewMixture(weights); err != nil {
		return nil, err
	} else {
		if len(edist) == 0 {
			edist = make([]VectorPdf, mixture.NComponents())
		} else {
			if mixture.NComponents() != len(edist) {
				return nil, fmt.Errorf("invalid number of emission distributions")
			}
			for i := 1; i < len(edist); i++ {
				if edist[0].Dim() != edist[i].Dim() {
					return nil, fmt.Errorf("emission distributions have inconsistent dimensions")
				}
			}
		}
		return &Mixture{*mixture, edist}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) Clone() *Mixture {
	edist := make([]VectorPdf, len(obj.Edist))
	for i := 0; i < len(obj.Edist); i++ {
		if obj.Edist[i] != nil {
			edist[i] = obj.Edist[i].CloneVectorPdf()
		}
	}
	return &Mixture{*obj.Mixture.Clone(), edist}
}

func (obj *Mixture) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) Dim() int {
	if len(obj.Edist) == 0 || obj.Edist[0] == nil {
		return 0
	} else {
		return obj.Edist[0].Dim()
	}
}

func (obj *Mixture) LogPdf(r Scalar, x ConstVector) error {
	return obj.Mixture.LogPdf(r, MixtureDataRecord{obj.Edist, x})
}

func (obj *Mixture) Likelihood(r Scalar, x ConstVector, states []int) error {
	return obj.Mixture.Likelihood(r, MixtureDataRecord{obj.Edist, x}, states)
}

func (obj *Mixture) Posterior(r Scalar, x ConstVector, states []int) error {
	return obj.Mixture.Posterior(r, MixtureDataRecord{obj.Edist, x}, states)
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) GetParameters() Vector {
	p := obj.Mixture.GetParameters()
	for i := 0; i < obj.NComponents(); i++ {
		p = p.AppendVector(obj.Edist[i].GetParameters())
	}
	return p
}

func (obj *Mixture) SetParameters(parameters Vector) error {
	n := obj.Mixture.GetParameters().Dim()
	obj.SetParameters(parameters.Slice(0, n))
	if err := obj.Mixture.SetParameters(parameters.Slice(0, n)); err != nil {
		return err
	}
	if parameters.Dim() > 0 {
		for i := 0; i < obj.NComponents(); i++ {
			n := obj.Edist[i].GetParameters().Dim()
			if err := obj.Edist[i].SetParameters(parameters.Slice(0, n)); err != nil {
				return err
			}
			parameters = parameters.Slice(n, parameters.Dim())
		}
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) String() string {
	var buffer bytes.Buffer

	fmt.Fprintf(&buffer, obj.Mixture.String())
	fmt.Fprintf(&buffer, "Emissions:\n")
	for i := 0; i < obj.NComponents(); i++ {
		fmt.Fprintf(&buffer, "-> %+v\n", obj.Edist[i].GetParameters())
	}
	return buffer.String()
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if err := obj.Mixture.ImportConfig(config, t); err != nil {
		return err
	}

	distributions := make([]VectorPdf, len(config.Distributions))
	for i := 0; i < len(config.Distributions); i++ {
		if tmp, err := ImportVectorPdfConfig(config.Distributions[i], t); err != nil {
			return err
		} else {
			distributions[i] = tmp
		}
	}
	obj.Edist = distributions

	return nil
}

func (obj *Mixture) ExportConfig() ConfigDistribution {

	distributions := make([]ConfigDistribution, len(obj.Edist))
	for i := 0; i < len(obj.Edist); i++ {
		distributions[i] = obj.Edist[i].ExportConfig()
	}
	config := obj.Mixture.ExportConfig()
	config.Name = "vector:mixture distribution"
	config.Distributions = distributions

	return config
}
