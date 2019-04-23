/* Copyright (C) 2015-2019 Philipp Benner
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

//import "fmt"
import "sort"

/* -------------------------------------------------------------------------- */

const vectorSparseIndexMax = int(^uint(0) >> 1)

type vectorSparseIndexSlice struct {
  values   []int
  isSorted   bool
}

func (obj *vectorSparseIndexSlice) sort() {
  if obj.isSorted == false {
    sort.Ints(obj.values)
    // remove revoked indices
    for i := len(obj.values)-1; i >= 0; i-- {
      if obj.values[i] == vectorSparseIndexMax {
        obj.values = obj.values[0:i]
      }
    }
    obj.isSorted = true
  }
}

func (obj *vectorSparseIndexSlice) insert(i int) {
  obj.values = append(obj.values, i)
  obj.isSorted = false
}

func (obj *vectorSparseIndexSlice) revoke(k int) {
  obj.values[k] = vectorSparseIndexMax
  obj.isSorted  = false
}

func (obj *vectorSparseIndexSlice) reverse() {
  for i := len(obj.values)/2-1; i >= 0; i-- {
    j := len(obj.values)-1-i
    obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
  }
}

func (obj *vectorSparseIndexSlice) find(i int) int {
  obj.sort()
  return sort.SearchInts(obj.values, i)
}

func (obj *vectorSparseIndexSlice) swap(i, j int) {
  i_k := obj.find(i)
  j_k := obj.find(j)
  obj.values[i_k], obj.values[j_k] = obj.values[j_k], obj.values[i_k]
}

func (obj vectorSparseIndexSlice) clone() vectorSparseIndexSlice {
  r := vectorSparseIndexSlice{}
  r.values = make([]int, len(obj.values))
  copy(r.values, obj.values)
  return r
}
