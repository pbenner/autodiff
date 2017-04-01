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

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestMgamma(t *testing.T) {
  if math.Abs(Mgamma(12, 2) - 8.418923e+14) > 1e10  ||
    (math.Abs(Mgamma(12, 3) - 9.597751e+21) > 1e15) {
    t.Error("Mgamma failed!")
  }
}

func TestMlgamma(t *testing.T) {
  if math.Abs(Mlgamma(12, 2) - math.Log(8.418923e+14)) > 1e-4  ||
    (math.Abs(Mlgamma(12, 3) - math.Log(9.597751e+21)) > 1e-4) {
    t.Error("Mlgamma failed!")
  }
}
