/* Copyright (C) 2015-2020 Philipp Benner
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

package gramSchmidt

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(test *testing.T) {

  t := NewFloat64(0.0)

  a := NewDenseFloat64Matrix([]float64{
    12, -51,   4,
     6, 167, -68,
    -4,  24, -41}, 3, 3)

  q, r, _ := Run(a)

  r1 := NewDenseFloat64Matrix([]float64{
     6.0/7.0, -69.0/175.0, -58.0/175.0,
     3.0/7.0, 158.0/175.0,   6.0/175.0,
    -2.0/7.0,   6.0/ 35.0, -33.0/ 35.0}, 3, 3)
  r2 := NewDenseFloat64Matrix([]float64{
    14,  21, -14,
     0, 175, -70,
     0,   0,  35}, 3, 3)

  if t.Mnorm(q.MsubM(r1, q)).GetFloat64() > 1e-8 {
    test.Error("Gram-Schmidt failed!")
  }
  if t.Mnorm(r.MsubM(r2, r)).GetFloat64() > 1e-8 {
    test.Error("Gram-Schmidt failed!")
  }  
}
