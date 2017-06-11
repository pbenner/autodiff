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

package distribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestGammaDistribution1(t *testing.T) {
  alpha  := NewBareReal(2.0)
  beta   := NewBareReal(4.0)
  dist,_ := NewGammaDistribution(alpha, beta)

  x := NewVector(BareRealType, []float64{0.2})
  y := NewReal(0.0)

  dist.LogPdf(y, x)
  r := NewBareReal(0.3631508)

  if Abs(Sub(r, y)).GetValue() > 1e-4 {
    t.Error("Gamma LogPdf failed")
  }
}
