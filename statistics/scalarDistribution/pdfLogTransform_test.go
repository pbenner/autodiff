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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func TestPdfLogTransform1(t *testing.T) {

	var d ScalarPdf

	mu := NewReal(2.0)
	sigma := NewReal(3.0)

	d, _ = NewNormalDistribution(mu, sigma)
	d, _ = NewPdfLogTransform(d, 0.0)

	x := NewReal(4.0)
	r := NewReal(0.0)

	d.LogPdf(r, x)

	if math.Abs(r.GetValue() - -3.424769) > 1e-4 {
		t.Error("test failed")
	}
}
