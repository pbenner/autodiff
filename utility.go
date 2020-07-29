/* Copyright (C) 2015-2020 Philipp Benner
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

/* -------------------------------------------------------------------------- */

type sortIntConstFloat64 struct {
  a []int
  b []float64
}

func (obj sortIntConstFloat64) Len() int {
  return len(obj.a)
}

func (obj sortIntConstFloat64) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstFloat64) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstFloat32 struct {
  a []int
  b []float32
}

func (obj sortIntConstFloat32) Len() int {
  return len(obj.a)
}

func (obj sortIntConstFloat32) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstFloat32) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstInt64 struct {
  a []int
  b []int64
}

func (obj sortIntConstInt64) Len() int {
  return len(obj.a)
}

func (obj sortIntConstInt64) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstInt64) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstInt32 struct {
  a []int
  b []int32
}

func (obj sortIntConstInt32) Len() int {
  return len(obj.a)
}

func (obj sortIntConstInt32) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstInt32) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstInt16 struct {
  a []int
  b []int16
}

func (obj sortIntConstInt16) Len() int {
  return len(obj.a)
}

func (obj sortIntConstInt16) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstInt16) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstInt8 struct {
  a []int
  b []int8
}

func (obj sortIntConstInt8) Len() int {
  return len(obj.a)
}

func (obj sortIntConstInt8) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstInt8) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

/* -------------------------------------------------------------------------- */

type sortIntConstInt struct {
  a []int
  b []int
}

func (obj sortIntConstInt) Len() int {
  return len(obj.a)
}

func (obj sortIntConstInt) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj sortIntConstInt) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}
