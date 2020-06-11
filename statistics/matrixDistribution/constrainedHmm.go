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

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Chmm struct {
	Hmm
}

/* -------------------------------------------------------------------------- */

func NewConstrainedHmm(pi Vector, tr Matrix, stateMap []int, edist []VectorPdf, constraints []generic.EqualityConstraint) (*Chmm, error) {
	return newConstrainedHmm(pi, tr, stateMap, edist, constraints, false)
}

func newConstrainedHmm(pi Vector, tr Matrix, stateMap []int, edist []VectorPdf, constraints []generic.EqualityConstraint, isLog bool) (*Chmm, error) {
	p, err := generic.NewHmmProbabilityVector(pi, isLog)
	if err != nil {
		return nil, err
	}
	t, err := generic.NewChmmTransitionMatrix(tr, constraints, isLog)
	if err != nil {
		return nil, err
	}
	if hmm, err := generic.NewHmm(p, t, stateMap); err != nil {
		return nil, err
	} else {
		if len(edist) == 0 {
			edist = make([]VectorPdf, hmm.NEDists())
		} else {
			if hmm.NEDists() != len(edist) {
				return nil, fmt.Errorf("invalid number of emission distributions")
			}
		}
		return &Chmm{Hmm{*hmm, edist}}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *Chmm) ImportConfig(config ConfigDistribution, t ScalarType) error {
	hmm := Hmm{}
	if err := hmm.ImportConfig(config, t); err != nil {
		return err
	}
	c, ok := config.GetNamedParametersAsNestedInts("Constraints")
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	constraints := []generic.EqualityConstraint{}
	if a1, ok1 := c.([]interface{}); ok1 {
		for i := 0; i < len(a1); i++ {
			if a2, ok2 := a1[i].([]interface{}); ok2 {
				constraint := generic.EqualityConstraint{}
				for j := 0; j < len(a2); j++ {
					if a3, ok3 := a2[j].([]interface{}); ok3 {
						c1 := 0
						c2 := 0
						if len(a3) != 2 {
							goto err
						}
						if a, ok := a3[0].(int); ok {
							c1 = a
						} else {
							goto err
						}
						if a, ok := a3[1].(int); ok {
							c2 = a
						} else {
							goto err
						}
						constraint = append(constraint, [2]int{c1, c2})
					}
				}
				constraints = append(constraints, constraint)
			}
		}
	}
	// get hierarchical transition matrix
	if r, err := newConstrainedHmm(hmm.Pi, hmm.Tr, hmm.StateMap, hmm.Edist, constraints, true); err != nil {
		return err
	} else {
		*obj = *r
	}
	return nil
err:
	return fmt.Errorf("invalid config")
}

func (obj *Chmm) ExportConfig() ConfigDistribution {
	configHmm := obj.Hmm.ExportConfig()
	parametersHmm := configHmm.Parameters.(struct {
		Pi          []float64
		Tr          []float64
		StateMap    []int
		N           int
		StartStates []int
		FinalStates []int
	})

	parameters := struct {
		Pi          []float64
		Tr          []float64
		StateMap    []int
		N           int
		StartStates []int
		FinalStates []int
		Constraints [][][2]int
	}{}

	constraints := obj.Tr.(generic.ChmmTransitionMatrix).GetConstraints()

	parameters.Pi = parametersHmm.Pi
	parameters.Tr = parametersHmm.Tr
	parameters.StateMap = parametersHmm.StateMap
	parameters.N = parametersHmm.N
	parameters.StartStates = parametersHmm.StartStates
	parameters.FinalStates = parametersHmm.FinalStates
	parameters.Constraints = make([][][2]int, len(constraints))

	for i := 0; i < len(constraints); i++ {
		parameters.Constraints[i] = constraints[i]
	}

	return NewConfigDistribution("matrix:constrained hmm distribution", parameters, configHmm.Distributions...)
}
