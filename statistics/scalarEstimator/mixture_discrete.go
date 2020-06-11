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

package scalarEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type DiscreteMixtureEstimator struct {
	MixtureEstimator
}

/* -------------------------------------------------------------------------- */

func NewDiscreteMixtureEstimator(weights []float64, estimators []ScalarEstimator, epsilon float64, maxSteps int, args ...interface{}) (*DiscreteMixtureEstimator, error) {
	if r, err := NewMixtureEstimator(weights, estimators, epsilon, maxSteps, args...); err != nil {
		return nil, err
	} else {
		return &DiscreteMixtureEstimator{*r}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj *DiscreteMixtureEstimator) CloneScalarEstimator() ScalarEstimator {
	return &DiscreteMixtureEstimator{*obj.MixtureEstimator.Clone()}
}

/* -------------------------------------------------------------------------- */

func (obj *DiscreteMixtureEstimator) SetData(x ConstVector, n int) error {
	if data, err := NewMixtureSummarizedDataSet(obj.ScalarType(), x, obj.mixture1.NComponents()); err != nil {
		return err
	} else {
		obj.data = data
	}
	for i, estimator := range obj.estimators {
		// set data
		if err := estimator.SetData(obj.data.GetData(), n); err != nil {
			return err
		}
		// initialize distribution
		if d, err := estimator.GetEstimate(); err != nil {
			return err
		} else {
			obj.mixture1.Edist[i] = d.CloneScalarPdf()
			obj.mixture2.Edist[i] = d.CloneScalarPdf()
			obj.mixture3.Edist[i] = d.CloneScalarPdf()
		}
	}
	return nil
}
