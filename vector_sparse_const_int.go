/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2020 Philipp Benner
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
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
package autodiff
/* -------------------------------------------------------------------------- */
import "fmt"
import "bytes"
import "encoding/json"
import "sort"
/* vector type declaration
 * -------------------------------------------------------------------------- */
type SparseConstIntVector struct {
  values []int
  indices []int
  idxmap map[int]int
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
func UnsafeSparseConstIntVector(indices []int, values []int, n int) SparseConstIntVector {
  r := nilSparseConstIntVector(n)
  r.indices = indices
  r.values = values
  return r
}
// Allocate a new vector. Scalars are set to the given values.
func NewSparseConstIntVector(indices []int, values []int, n int) SparseConstIntVector {
  if len(indices) != len(values) {
    panic("invalid number of indices")
  }
  sort.Sort(sortIntConstInt{indices, values})
  r := nilSparseConstIntVector(n)
  r.indices = indices[0:0]
  r.values = make([]int, 0, len(values))
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if values[i] != 0.0 {
      r.values = append(r.values, values[i])
      r.indices = append(r.indices, k)
    }
  }
  return r
}
func nilSparseConstIntVector(n int) SparseConstIntVector {
  r := SparseConstIntVector{}
  r.n = n
  // create map here so that no pointer receivers are needed
  r.idxmap = make(map[int]int)
  return r
}
// Convert vector type.
func AsSparseConstIntVector(v ConstVector) SparseConstIntVector {
  switch v_ := v.(type) {
  case SparseConstIntVector:
    return v_
  }
  indices := []int{}
  values := []int{}
  n := v.Dim()
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    indices = append(indices, it.Index())
    values = append(values, it.GetConst().GetInt())
  }
  return NewSparseConstIntVector(indices, values, n)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) Clone() SparseConstIntVector {
  r := nilSparseConstIntVector(obj.n)
  r.indices = obj.indices
  r.values = obj.values
  return r
}
/* methods specific to this type
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) GetSparseIndices() []int {
  return obj.indices
}
func (obj SparseConstIntVector) GetSparseValues() []int {
  return obj.values
}
func (obj SparseConstIntVector) First() (int, ConstInt) {
  return obj.indices[0], ConstInt(obj.values[0])
}
func (obj SparseConstIntVector) Last() (int, ConstInt) {
  i := len(obj.indices) - 1
  return obj.indices[i], ConstInt(obj.values[i])
}
func (obj SparseConstIntVector) CreateIndex() {
  if len(obj.idxmap) == 0 {
    for i, k := range obj.indices {
      obj.idxmap[k] = i
    }
  }
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj SparseConstIntVector) Dim() int {
  return obj.n
}
func (obj SparseConstIntVector) Int8At(i int) int8 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int8(obj.values[k])
  } else {
    return int8(0)
  }
}
func (obj SparseConstIntVector) Int16At(i int) int16 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int16(obj.values[k])
  } else {
    return int16(0)
  }
}
func (obj SparseConstIntVector) Int32At(i int) int32 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int32(obj.values[k])
  } else {
    return int32(0)
  }
}
func (obj SparseConstIntVector) Int64At(i int) int64 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int64(obj.values[k])
  } else {
    return int64(0)
  }
}
func (obj SparseConstIntVector) IntAt(i int) int {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int(obj.values[k])
  } else {
    return int(0)
  }
}
func (obj SparseConstIntVector) Float32At(i int) float32 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return float32(obj.values[k])
  } else {
    return float32(0)
  }
}
func (obj SparseConstIntVector) Float64At(i int) float64 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return float64(obj.values[k])
  } else {
    return float64(0)
  }
}
func (obj SparseConstIntVector) ConstAt(i int) ConstScalar {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return ConstFloat64(obj.values[k])
  } else {
    return ConstFloat64(0.0)
  }
}
func (obj SparseConstIntVector) ConstSlice(i, j int) ConstVector {
  if i == 0 {
    k1 := 0
    k2 := sort.SearchInts(obj.indices, j)
    r := nilSparseConstIntVector(j)
    r.values = obj.values [k1:k2]
    r.indices = obj.indices[k1:k2]
    return r
  } else {
    k1 := sort.SearchInts(obj.indices, i)
    k2 := sort.SearchInts(obj.indices, j)
    r := nilSparseConstIntVector(j-i)
    r.values = obj.values[k1:k2]
    r.indices = make([]int, k2-k1)
    for k := k1; k < k2; k++ {
      r.indices[k-k1] = obj.indices[k] - i
    }
    return r
  }
}
func (obj SparseConstIntVector) AsConstMatrix(n, m int) ConstMatrix {
  panic("not implemented")
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, ConstInt(v))
  }
  return r
}
func (obj SparseConstIntVector) ElementType() ScalarType {
  return ConstIntType
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj SparseConstIntVector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj SparseConstIntVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj SparseConstIntVector) ITERATOR() *SparseConstIntVectorIterator {
  r := SparseConstIntVectorIterator{0, obj}
  return &r
}
func (obj SparseConstIntVector) ITERATOR_FROM(i int) *SparseConstIntVectorIterator {
  k := 0
  for j, idx := range obj.indices {
    if idx >= i {
      k = j
      break
    }
  }
  r := SparseConstIntVectorIterator{k, obj}
  return &r
}
func (obj SparseConstIntVector) JOINT_ITERATOR(b ConstVector) *SparseConstIntVectorJointIterator {
  r := SparseConstIntVectorJointIterator{}
  r.it1 = obj.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) String() string {
  var buffer bytes.Buffer
  buffer.WriteString(fmt.Sprintf("%d:[", obj.n))
  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(", ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%v", it.Index(), it.GetConst()))
  }
  buffer.WriteString("]")
  return buffer.String()
}
func (obj SparseConstIntVector) Table() string {
  var buffer bytes.Buffer
  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(" ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%v", it.Index(), it.GetConst()))
  }
  if len(obj.indices) > 0 {
    if i := obj.indices[len(obj.indices)-1]; i != obj.n-1 {
      buffer.WriteString(fmt.Sprintf(" %d:%v", i, int(0)))
    }
  }
  return buffer.String()
}
/* json
 * -------------------------------------------------------------------------- */
func (obj SparseConstIntVector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []int{}
  r := struct{
    Index []int
    Value []int
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetInt())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseConstIntVectorIterator struct {
  i int
  v SparseConstIntVector
}
func (obj *SparseConstIntVectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *SparseConstIntVectorIterator) GET() ConstInt {
  return ConstInt(obj.v.values[obj.i])
}
func (obj *SparseConstIntVectorIterator) Ok() bool {
  return obj.i < len(obj.v.indices)
}
func (obj *SparseConstIntVectorIterator) Index() int {
  return obj.v.indices[obj.i]
}
func (obj *SparseConstIntVectorIterator) Next() {
  obj.i += 1
}
func (obj *SparseConstIntVectorIterator) Clone() *SparseConstIntVectorIterator {
  return &SparseConstIntVectorIterator{obj.i, obj.v}
}
func (obj *SparseConstIntVectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseConstIntVectorIterator{obj.i, obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseConstIntVectorJointIterator struct {
  it1 *SparseConstIntVectorIterator
  it2 VectorConstIterator
  idx int
  s1 ConstInt
  s2 ConstScalar
}
func (obj *SparseConstIntVectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseConstIntVectorJointIterator) Ok() bool {
  return !(obj.s1.GetInt() == int(0)) ||
         !(obj.s2.GetInt() == int(0))
}
func (obj *SparseConstIntVectorJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = ConstInt(0)
  obj.s2 = ConstInt(0)
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1 = ConstInt(0)
      obj.s2 = obj.it2.GetConst()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1 != ConstInt(0) {
    obj.it1.Next()
  }
  if obj.s2 != ConstInt(0) {
    obj.it2.Next()
  } else {
    obj.s2 = ConstInt(0.0)
  }
}
func (obj *SparseConstIntVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseConstIntVectorJointIterator) GET() (ConstInt, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseConstIntVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  r := SparseConstIntVectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
/* math
 * -------------------------------------------------------------------------- */
// Test if elements in a equal elements in b.
func (a SparseConstIntVector) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("Equals(): Vector dimensions do not match!")
  }
  for it := a.ConstJointIterator(b); it.Ok(); it.Next() {
    s1, s2 := it.GetConst()
    if !s1.Equals(s2, epsilon) {
      return false
    }
  }
  return true
}
