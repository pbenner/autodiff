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
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/threadpool"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test2(t *testing.T) {
	// Hmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	d1, _ := NewNormalEstimator(1.0, 2.0, 1e-6)
	d2, _ := NewNormalEstimator(1.0, 3.0, 1e-6)
	e1, _ := NewLogTransformEstimator(d1, 0.0)
	e2, _ := NewLogTransformEstimator(d2, 0.0)

	// observations
	//////////////////////////////////////////////////////////////////////////////
	x := NewVector(RealType, []float64{
		//> rlnorm(25, 1, 2)
		03.7046313, 15.2576178, 0.6321346, 2.4182017, 1.3459458,
		27.4053370, 1.6910290, 2.5277558, 5.1212931, 11.9720053,
		01.1407313, 0.1404276, 9.4357250, 52.5796490, 24.3452806,
		03.8663017, 37.6600575, 3.4580107, 223.1953995, 3.6914441,
		17.6846846, 35.5848270, 0.1735938, 1218.8903756, 2.8297499,
		//> rlnorm(25, 2, 1)
		03.615067, 20.131041, 12.810724, 5.058049, 4.720553, 5.530191, 12.597413,
		04.824556, 34.410099, 6.655225, 5.092371, 21.789605, 9.471491, 6.131201,
		14.517041, 8.380483, 19.718602, 13.069624, 7.223624, 1.635017, 2.081164,
		07.086723, 12.181270, 7.472553, 10.356920})

	r := NewVector(RealType, []float64{
		-1.655838e+02, -1.224244e-72, -1.196199e-11, -2.514928e+01, -3.236638e+00,
		-4.008872e-02, 2.086974e+00, 7.035202e-01, 1.891452e+00, 1.992563e+00})

	// Baum-Welch
	//////////////////////////////////////////////////////////////////////////////
	if estimator, err := vectorEstimator.NewHmmEstimator(pi, tr, nil, nil, nil, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
		t.Error(err)
	} else {
		if err = estimator.EstimateOnData([]ConstVector{x}, nil, threadpool.New(2, 100)); err != nil {
			t.Error(err)
			return
		}
		hmm, _ := estimator.GetEstimate()

		if Vnorm(VsubV(r, hmm.GetParameters())).GetValue() > 1e-2 {
			t.Error("test failed")
		}
	}
}
