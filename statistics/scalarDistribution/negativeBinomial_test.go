/* Copyright (C) 2016-2020 Philipp Benner
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
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestNegativeBinomial(t *testing.T) {
  d, _ := NewNegativeBinomialDistribution(NewFloat64(3), NewFloat64(0.3))
  x := NewFloat64(12.0)
  r := NewFloat64(0.0)

  if err := d.LogPdf(r, x); err != nil {
    t.Error(err)
  }
  if math.Abs(r.GetFloat64() - -11.00684) > 1e-4 {
    t.Error("test failed")
  }
}
