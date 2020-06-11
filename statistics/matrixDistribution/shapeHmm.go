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

package matrixDistribution

/* -------------------------------------------------------------------------- */

import "fmt"
import "bytes"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ShapeHmm struct {
	generic.Hmm
	Edist []MatrixPdf
}

/* -------------------------------------------------------------------------- */

func NewShapeHmm(pi Vector, tr Matrix, stateMap []int, edist []MatrixPdf) (*ShapeHmm, error) {
	p, err := generic.NewHmmProbabilityVector(pi, false)
	if err != nil {
		return nil, err
	}
	t, err := generic.NewHmmTransitionMatrix(tr, false)
	if err != nil {
		return nil, err
	}
	if hmm, err := generic.NewHmm(p, t, stateMap); err != nil {
		return nil, err
	} else {
		if len(edist) == 0 {
			edist = make([]MatrixPdf, hmm.NEDists())
		} else {
			if hmm.NEDists() != len(edist) {
				return nil, fmt.Errorf("invalid number of emission distributions")
			}
			for i := 1; i < len(edist); i++ {
				n1, m1 := edist[0].Dims()
				n2, m2 := edist[i].Dims()
				if n1 != n2 || m1 != m2 {
					return nil, fmt.Errorf("emission distributions have inconsistent dimensions")
				}
			}
		}
		return &ShapeHmm{*hmm, edist}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmm) Clone() *ShapeHmm {
	edist := make([]MatrixPdf, len(obj.Edist))
	for i := 0; i < len(obj.Edist); i++ {
		edist[i] = obj.Edist[i].CloneMatrixPdf()
	}
	return &ShapeHmm{*obj.Hmm.Clone(), edist}
}

func (obj *ShapeHmm) CloneMatrixPdf() MatrixPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmm) Dims() (int, int) {
	if len(obj.Edist) == 0 || obj.Edist[0] == nil {
		return 0, -1
	} else {
		n, _ := obj.Edist[0].Dims()
		return n, -1
	}
}

func (obj *ShapeHmm) LogPdf(r Scalar, x ConstMatrix) error {
	return obj.Hmm.LogPdf(r, ShapeHmmDataRecord{obj.Edist, x})
}

func (obj *ShapeHmm) Posterior(r Scalar, x Matrix, states [][]int) error {
	return obj.Hmm.Posterior(r, ShapeHmmDataRecord{obj.Edist, x}, states)
}

func (obj *ShapeHmm) PosteriorMarginals(x Matrix) ([]Vector, error) {
	return obj.Hmm.PosteriorMarginals(ShapeHmmDataRecord{obj.Edist, x})
}

func (obj *ShapeHmm) Viterbi(x Matrix) ([]int, error) {
	return obj.Hmm.Viterbi(ShapeHmmDataRecord{obj.Edist, x})
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmm) GetParameters() Vector {
	p := obj.Hmm.GetParameters()
	for i := 0; i < obj.NEDists(); i++ {
		p = p.AppendVector(obj.Edist[i].GetParameters())
	}
	return p
}

func (obj *ShapeHmm) SetParameters(parameters Vector) error {
	n := obj.Hmm.GetParameters().Dim()
	if err := obj.Hmm.SetParameters(parameters.Slice(0, n)); err != nil {
		return err
	}
	parameters = parameters.Slice(n, parameters.Dim())
	if parameters.Dim() > 0 {
		for i := 0; i < obj.NEDists(); i++ {
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

func (obj *ShapeHmm) String() string {
	var buffer bytes.Buffer

	fmt.Fprintf(&buffer, obj.Hmm.String())
	fmt.Fprintf(&buffer, "Emissions:\n")
	for i := 0; i < obj.NEDists(); i++ {
		fmt.Fprintf(&buffer, "-> %+v\n", obj.Edist[i].GetParameters())
	}
	return buffer.String()
}

/* -------------------------------------------------------------------------- */

func (obj *ShapeHmm) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if err := obj.Hmm.ImportConfig(config, t); err != nil {
		return err
	}

	distributions := make([]MatrixPdf, len(config.Distributions))
	for i := 0; i < len(config.Distributions); i++ {
		if tmp, err := ImportMatrixPdfConfig(config.Distributions[i], t); err != nil {
			return err
		} else {
			distributions[i] = tmp
		}
	}
	obj.Edist = distributions

	return nil
}

func (obj *ShapeHmm) ExportConfig() ConfigDistribution {

	distributions := make([]ConfigDistribution, len(obj.Edist))
	for i := 0; i < len(obj.Edist); i++ {
		distributions[i] = obj.Edist[i].ExportConfig()
	}
	config := obj.Hmm.ExportConfig()
	config.Name = "matrix:shape hmm distribution"
	config.Distributions = distributions

	return config
}
