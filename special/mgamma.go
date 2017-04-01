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

// multivariate gamma function
func Mgamma(x float64, k int) float64 {
  result := math.Pow(math.Pi, float64(k*(k-1.0))/4.0)
  for i := 1; i <= k; i += 1 {
    result *= math.Gamma((2.0*x+1.0-float64(i))/2.0)
  }
  return result
}

func Mlgamma(x float64, k int) float64 {
  result := float64(k*(k-1.0))/4.0*math.Log(math.Pi)
  for i := 1; i <= k; i += 1 {
    v, _ := math.Lgamma((2.0*x+1.0-float64(i))/2.0)
    result += v
  }
  return result
}
