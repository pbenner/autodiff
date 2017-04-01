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

package mi

/* -------------------------------------------------------------------------- */

import   "math"

/* mutual information
 * -------------------------------------------------------------------------- */

func MI(channel [][]float64, px []float64) float64 {
  n := len(channel)
  m := len(channel[0])
  if len(px) != n {
    panic("Input vector has invalid dimension!")
  }
  // compute p(y) from p(y|x)*p(x)
  py := make([]float64, m)
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      py[j] += channel[i][j]*px[i]
    }
  }
  // compute mutual information
  mi := 0.0
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      if channel[i][j] > 0.0 {
        mi += channel[i][j]*px[i]*math.Log2(channel[i][j]/py[j])
      }
    }
  }
  return mi
}
