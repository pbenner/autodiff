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

import "math"

/* -------------------------------------------------------------------------- */

type ContinuedFraction interface {
  Eval() (float64, float64)
}

/* -------------------------------------------------------------------------- */

func EvalContinuedFraction(fraction ContinuedFraction, factor float64, max_terms int) float64 {

  tiny := math.SmallestNonzeroFloat64
  a, b := fraction.Eval()

  f  := b
  a0 := a

  if f == 0.0 {
    f = tiny
  }
  C := f
  D := 0.0
  delta := 0.0

  for i := 0; i < max_terms; i++ {
    a, b = fraction.Eval()
    D = b + a*D;
    if D == 0.0 {
      D = tiny
    }
    C = b + a/C
    if C == 0 {
      C = tiny
    }
    D = 1.0/D
    delta = C*D
    f = f*delta
    if math.Abs(delta - 1.0) <= factor {
      break
    }
  }
  return a0/f
}
