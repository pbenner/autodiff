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

package autodiff

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

func logAdd(a, b float64) float64 {
  if a > b {
    // swap
    a, b = b, a
  }
  if math.IsInf(a, -1) {
    return b
  }
  return b + math.Log1p(math.Exp(a-b))
}

func logSub(a, b float64) float64 {
  if math.IsInf(b, -1) {
    return a
  }
  return a + math.Log1p(-math.Exp(b-a))
}
