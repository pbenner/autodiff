/* Copyright (C) 2018 Philipp Benner
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
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NilEstimator struct {
	ScalarPdf
}

/* -------------------------------------------------------------------------- */

func (obj NilEstimator) CloneScalarEstimator() ScalarEstimator {
	return NilEstimator{obj.ScalarPdf.CloneScalarPdf()}
}

/* -------------------------------------------------------------------------- */

func (NilEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	return nil
}

func (NilEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	return nil
}

func (NilEstimator) SetData(x ConstVector, n int) error {
	return nil
}

func (obj NilEstimator) GetEstimate() (ScalarPdf, error) {
	return obj.ScalarPdf, nil
}
