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

const M_SQRTPI      = 1.77245385090551602729816748334
const M_PI          = 3.14159265358979323846264338328
const M_ROOT_TWO_PI = 2.506628274631000502415765284811045253

/* -------------------------------------------------------------------------- */

var MaxLogFloat64       float64
var MinLogFloat64       float64
var EpsilonFloat64      float64
var PrecisionFloat64    int
var SeriesIterationsMax int
var MaxFactorial        int

/* -------------------------------------------------------------------------- */

func init() {
  MaxLogFloat64       = math.Floor(math.Log(math.MaxFloat64))
  MinLogFloat64       = math.Ceil (math.Log(math.SmallestNonzeroFloat64))
  EpsilonFloat64      = math.Nextafter(1.0,2.0)-1.0
  PrecisionFloat64    = 53
  SeriesIterationsMax = 1000000
  MaxFactorial        = 170
}
