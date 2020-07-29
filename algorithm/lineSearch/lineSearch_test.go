/* Copyright (C) 2017-2020 Philipp Benner
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

package lineSearch

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestLineSearch(test *testing.T) {

  f := func(x ConstScalar) MagicScalar {
    a := NullReal64()
    b := NullReal64()
    c := NullReal64()
    t := NullReal64()
    a.Sub(x, ConstFloat64(3.0))
    b.Pow(x, ConstFloat64(3.0))
    c.Pow(t.Sub(x, ConstFloat64(6.0)), ConstFloat64(4.0))
    t.Mul(t.Mul(a, b), c)
    return t
  }
  g := func(alpha ConstScalar) (MagicScalar, error) {
    t := NullReal64()
    return f(t.Add(ConstFloat64(1.7), alpha)), nil
  }
  // hook := func(x, y, g Scalar) bool {
  //   fmt.Println("x:", x)
  //   fmt.Println("y:", y)
  //   return false
  // }
  x, err := Run(g, Float64Type)

  if err != nil {
    test.Error(err)
  } else {
    if math.Abs(x.GetFloat64() - 4.381409e-02) > 1e-6 {
      test.Error("TestLineSearch failed")
    }
  }
}
