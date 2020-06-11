/* Copyright (C) 2015 Philipp Benner
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

package blahut

/* -------------------------------------------------------------------------- */

import "testing"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestBlahut(t *testing.T) {

	channel := NewMatrix(RealType, 3, 3,
		flatten([][]float64{
			{0.60, 0.30, 0.10},
			{0.70, 0.10, 0.20},
			{0.50, 0.05, 0.45}}))

	// initial value
	px0 := NewVector(RealType,
		[]float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0})
	// fixed point
	pxstar := []float64{0.501735, 0.0, 0.498265}

	pxn := Run(channel, px0, 1000)

	for i := 0; i < px0.Dim(); i++ {
		if math.Abs(pxn.At(i).GetValue()-pxstar[i]) > 1e-5 {
			t.Error("Blahut test failed!")
		}
	}
}

func TestBlahutNaive(t *testing.T) {

	channel := [][]float64{
		{0.60, 0.30, 0.10},
		{0.70, 0.10, 0.20},
		{0.50, 0.05, 0.45}}

	// initial value
	px0 := []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}
	// fixed point
	pxstar := []float64{0.501735, 0.0, 0.498265}

	pxn := RunNaive(channel, px0, 1000)

	for i := 0; i < len(px0); i++ {
		if math.Abs(pxn[i]-pxstar[i]) > 1e-5 {
			t.Error("Blahut test failed!")
		}
	}
}
