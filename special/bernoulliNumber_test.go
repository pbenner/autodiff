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

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestBernoulliNumber(t *testing.T) {

  r := []float64{
     1.000000,
     0.500000,
     0.166666,
     0.0,
    -0.033333,
     0.0,
     0.023809,
     0.0,
    -0.033333,
     0.0,
     0.075757,
     0.0,
    -0.253113,
     0.0,
     1.166666,
     0.0,
    -7.092156,
     0.0,
     54.971177,
     0.0,
    -529.124242,
     0.0,
     6192.123188,
     0.0,
    -86580.253113 }

  for i := 0; i < len(r); i++ {
    if math.Abs(BernoulliNumber(i) - r[i]) > 1e-3 {
      t.Error("BernoulliNumber() failed!")
    }
  }
}
