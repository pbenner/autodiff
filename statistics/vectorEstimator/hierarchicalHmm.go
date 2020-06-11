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

package vectorEstimator

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func NewHierarchicalHmmEstimator(pi Vector, tr Matrix, stateMap, startStates, finalStates []int, tree generic.HmmNode, estimators []ScalarEstimator, epsilon float64, maxSteps int, args ...interface{}) (*HmmEstimator, error) {
	if hmm, err := vectorDistribution.NewHierarchicalHmm(pi, tr, stateMap, nil, tree); err != nil {
		return nil, err
	} else {
		if err := hmm.SetStartStates(startStates); err != nil {
			return nil, err
		}
		if err := hmm.SetFinalStates(finalStates); err != nil {
			return nil, err
		}
		if hmm.NEDists() > 0 && len(estimators) != hmm.NEDists() {
			return nil, fmt.Errorf("invalid number of estimators")
		}
		for i, estimator := range estimators {
			// initialize distribution
			if hmm.Edist[i] == nil {
				if d, err := estimator.GetEstimate(); err != nil {
					return nil, err
				} else {
					hmm.Edist[i] = d
				}
			}
		}
		// initialize estimators with data
		r := HmmEstimator{}
		r.hmm1 = hmm.Clone()
		r.hmm2 = hmm.Clone()
		r.hmm3 = hmm.Clone()
		r.estimators = estimators
		r.epsilon = epsilon
		r.maxSteps = maxSteps
		r.args = args
		r.OptimizeEmissions = true
		r.OptimizeTransitions = true
		return &r, nil
	}
}
