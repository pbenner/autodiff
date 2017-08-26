/* Copyright (C) 2017 Philipp Benner
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

package givensRotation

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

// Compute c and s such that
// [  c  s ]^T  [ a ]  =  [ r ]
// [ -s  c ]    [ b ]  =  [ 0 ]
//
// where c = cos(theta)
//       s = sin(theta)
//
// => c =  a / sqrt(a^2 + b^2)
//    s = -b / sqrt(a^2 + b^2)

func Run(a, b, c, s Scalar) {
  c1 := BareReal(1.0)

  if b.GetValue() == 0.0 {
    c.SetValue(1.0)
    s.SetValue(0.0)
  } else {
    if math.Abs(b.GetValue()) > math.Abs(a.GetValue()) {
      c.Div(a, b)
      c.Neg(c)

      s.Mul(c, c)
      s.Add(s, &c1)
      s.Sqrt(s)
      s.Div(&c1, s)

      c.Mul(c, s)
    } else {
      s.Div(b, a)
      s.Neg(s)

      c.Mul(s, s)
      c.Add(c, &c1)
      c.Sqrt(c)
      c.Div(&c1, c)

      s.Mul(s, c)
    }
  }
}
