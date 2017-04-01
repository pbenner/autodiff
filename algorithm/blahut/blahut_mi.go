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

package blahut

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func mi_hook(result *Scalar, p Vector, J Scalar) bool {
  *result = J
  return false
}

/* mutual information
 * -------------------------------------------------------------------------- */

func MI(channel Matrix, p Vector) Scalar {
  // resulting mutual information
  var result Scalar
  // create hook
  h := func(p Vector, J Scalar) bool { return mi_hook(&result, p, J) }
  // call blahut for one iteration
  Run(channel, p, 1, Hook{h})

  return result
}
