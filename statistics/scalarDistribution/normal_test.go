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

func TestNormal1(t *testing.T) {

  mu     := NewFloat64(3.0)
  sigma  := NewFloat64(math.Sqrt(2.0))

  normal, _ := NewNormalDistribution(mu, sigma)

  x := NewFloat64(2.2)
  y := NewFloat64(0.0)

  normal.LogCdf(y, x)

  if math.Abs(y.GetFloat64() - -1.2524496) > 1e-4 {
    t.Error("Normal LogCdf failed!")
  }
}
