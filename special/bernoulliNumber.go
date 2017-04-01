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

//import "math"
import "math/big"

/* -------------------------------------------------------------------------- */

func BernoulliNumber(n int) float64 {
  a := make([]big.Rat, n+1)
  r := &big.Rat{}
  for i := 0; i < n+1; i++ {
    a[i].SetFrac64(1, int64(i+1.0))
    for j := i; j > 0; j-- {
      d := &a[j-1]
      d.Mul(r.SetInt64(int64(j)), d.Sub(d, &a[j]))
    }
  }
  t, _ := a[0].Float64()
  return t
}
