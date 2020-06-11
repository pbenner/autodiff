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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestChiSquaredDistribution1(t *testing.T) {

	d, _ := NewChiSquaredDistribution(RealType, 2.0)

	x := NewReal(10.214)
	Variables(2, x)
	y := NewReal(0.0)
	d.LogPdf(y, x)

	if math.Abs(y.GetValue() - -5.800147) > 1e-4 {
		t.Error("Chi-Squared LogPdf failed!")
	}
	if math.Abs(y.GetDerivative(0) - -0.500000) > 1e-4 {
		t.Error("Chi-Squared LogPdf failed!")
	}
	if math.Abs(y.GetHessian(0, 0)-0.000000) > 1e-4 {
		t.Error("Chi-Squared LogPdf failed!")
	}
}

func TestChiSquaredDistribution2(t *testing.T) {

	d, _ := NewChiSquaredDistribution(RealType, 2.0)

	x := NewReal(4.817)
	Variables(2, x)
	y := NewReal(0.0)
	d.Cdf(y, x)

	if math.Abs(y.GetValue()-0.91005) > 1e-4 {
		t.Error("Chi-Squared Cdf failed!")
	}
	if math.Abs(y.GetDerivative(0)-0.0449751) > 1e-4 {
		t.Error("Chi-Squared Cdf failed!")
	}
	if math.Abs(y.GetHessian(0, 0) - -0.0224875) > 1e-4 {
		t.Error("Chi-Squared Cdf failed!")
	}
}
