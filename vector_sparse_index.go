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
  index       []int
  indexSorted   bool
}

func (obj vectorSparseIndexSlice) indexSort() {
  if obj.indexSorted == false {
    sort.Ints(obj.index)
    // remove revoked indices
    for i := len(obj.index)-1; i >= 0; i-- {
      if obj.index[i] == vectorSparseIndexMax {
        obj.index = obj.index[0:i]
      }
    }
    obj.indexSorted = true
  }
}

func (obj *vectorSparseIndexSlice) indexInsert(i int) {
  obj.index = append(obj.index, i)
  obj.indexSorted = false
}

func (obj vectorSparseIndexSlice) indexRevoke(k int) {
  obj.index[k] = vectorSparseIndexMax
  obj.indexSorted  = false
}

func (obj vectorSparseIndexSlice) indexReverse() {
  for i :=len(obj.index)/2-1; i >= 0; i-- {
    j := len(obj.index)-1-i
    obj.index[i], obj.index[j] = obj.index[j], obj.index[i]
  }
}

func (obj vectorSparseIndexSlice) indexFind(i int) int {
  obj.indexSort()
  return sort.SearchInts(obj.index, i)
}

func (obj vectorSparseIndexSlice) indexSwap(i, j int) {
  i_k := obj.indexFind(i)
  j_k := obj.indexFind(j)
  obj.index[i_k], obj.index[j_k] = obj.index[j_k], obj.index[i_k]
}

func (obj vectorSparseIndexSlice) indexCopy(src []int) {
  copy(obj.index, src)
  obj.indexSorted = false
}

func (obj vectorSparseIndexSlice) indexClone() vectorSparseIndexSlice {
  r := vectorSparseIndexSlice{}
  r.index = make([]int, len(obj.index))
  copy(r.index, obj.index)
  return r
}
