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
import "os"

/* -------------------------------------------------------------------------- */

func iMin(a, b int) int {
  if a < b {
    return a
  } else {
    return b
  }
}

func iMax(a, b int) int {
  if a > b {
    return a
  } else {
    return b
  }
}

func sign(a float64) int {
  if math.Signbit(a) {
    return -1
  } else {
    return 1
  }
}

func isGzip(filename string) (bool, error) {

  f, err := os.Open(filename)
  if err != nil {
    return false, err
  }
  defer f.Close()

  b := make([]byte, 2)
  n, err := f.Read(b)
  if err != nil {
    return false, err
  }

  if n == 2 && b[0] == 31 && b[1] == 139 {
    return true, nil
  }
  return false, nil
}
