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
type SparseConstFloat64Vector struct {
  values []float64
  indices []int
  idxmap map[int]int
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
func UnsafeSparseConstFloat64Vector(indices []int, values []float64, n int) SparseConstFloat64Vector {
  r := nilSparseConstFloat64Vector(n)
  r.indices = indices
  r.values = values
  return r
}
// Allocate a new vector. Scalars are set to the given values.
func NewSparseConstFloat64Vector(indices []int, values []float64, n int) SparseConstFloat64Vector {
  if len(indices) != len(values) {
    panic("invalid number of indices")
  }
  sort.Sort(sortIntConstFloat64{indices, values})
  r := nilSparseConstFloat64Vector(n)
  r.indices = indices[0:0]
  r.values = make([]float64, 0, len(values))
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
func nilSparseConstFloat64Vector(n int) SparseConstFloat64Vector {
  r := SparseConstFloat64Vector{}
  r.n = n
  // create map here so that no pointer receivers are needed
  r.idxmap = make(map[int]int)
  return r
}
// Convert vector type.
func AsSparseConstFloat64Vector(v ConstVector) SparseConstFloat64Vector {
  switch v_ := v.(type) {
  case SparseConstFloat64Vector:
    return v_
  }
  indices := []int{}
  values := []float64{}
  n := v.Dim()
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    indices = append(indices, it.Index())
    values = append(values, it.GetConst().GetFloat64())
  }
  return NewSparseConstFloat64Vector(indices, values, n)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) Clone() SparseConstFloat64Vector {
  r := nilSparseConstFloat64Vector(obj.n)
  r.indices = obj.indices
  r.values = obj.values
  return r
}
/* methods specific to this type
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) GetSparseIndices() []int {
  return obj.indices
}
func (obj SparseConstFloat64Vector) GetSparseValues() []float64 {
  return obj.values
}
func (obj SparseConstFloat64Vector) First() (int, ConstFloat64) {
  return obj.indices[0], ConstFloat64(obj.values[0])
}
func (obj SparseConstFloat64Vector) Last() (int, ConstFloat64) {
  i := len(obj.indices) - 1
  return obj.indices[i], ConstFloat64(obj.values[i])
}
func (obj SparseConstFloat64Vector) CreateIndex() {
  if len(obj.idxmap) == 0 {
    for i, k := range obj.indices {
      obj.idxmap[k] = i
    }
  }
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj SparseConstFloat64Vector) Dim() int {
  return obj.n
}
func (obj SparseConstFloat64Vector) Int8At(i int) int8 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int8(obj.values[k])
  } else {
    return int8(0)
  }
}
func (obj SparseConstFloat64Vector) Int16At(i int) int16 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int16(obj.values[k])
  } else {
    return int16(0)
  }
}
func (obj SparseConstFloat64Vector) Int32At(i int) int32 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int32(obj.values[k])
  } else {
    return int32(0)
  }
}
func (obj SparseConstFloat64Vector) Int64At(i int) int64 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int64(obj.values[k])
  } else {
    return int64(0)
  }
}
func (obj SparseConstFloat64Vector) IntAt(i int) int {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return int(obj.values[k])
  } else {
    return int(0)
  }
}
func (obj SparseConstFloat64Vector) Float32At(i int) float32 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return float32(obj.values[k])
  } else {
    return float32(0)
  }
}
func (obj SparseConstFloat64Vector) Float64At(i int) float64 {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return float64(obj.values[k])
  } else {
    return float64(0)
  }
}
func (obj SparseConstFloat64Vector) ConstAt(i int) ConstScalar {
  if len(obj.idxmap) == 0 {
    obj.CreateIndex()
  }
  if k, ok := obj.idxmap[i]; ok {
    return ConstFloat64(obj.values[k])
  } else {
    return ConstFloat64(0.0)
  }
}
func (obj SparseConstFloat64Vector) ConstSlice(i, j int) ConstVector {
  if i == 0 {
    k1 := 0
    k2 := sort.SearchInts(obj.indices, j)
    r := nilSparseConstFloat64Vector(j)
    r.values = obj.values [k1:k2]
    r.indices = obj.indices[k1:k2]
    return r
  } else {
    k1 := sort.SearchInts(obj.indices, i)
    k2 := sort.SearchInts(obj.indices, j)
    r := nilSparseConstFloat64Vector(j-i)
    r.values = obj.values[k1:k2]
    r.indices = make([]int, k2-k1)
    for k := k1; k < k2; k++ {
      r.indices[k-k1] = obj.indices[k] - i
    }
    return r
  }
}
func (obj SparseConstFloat64Vector) AsConstMatrix(n, m int) ConstMatrix {
  panic("not implemented")
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, ConstFloat64(v))
  }
  return r
}
func (obj SparseConstFloat64Vector) ElementType() ScalarType {
  return ConstFloat64Type
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj SparseConstFloat64Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj SparseConstFloat64Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj SparseConstFloat64Vector) ITERATOR() *SparseConstFloat64VectorIterator {
  r := SparseConstFloat64VectorIterator{0, obj}
  return &r
}
func (obj SparseConstFloat64Vector) ITERATOR_FROM(i int) *SparseConstFloat64VectorIterator {
  k := 0
  for j, idx := range obj.indices {
    if idx >= i {
      k = j
      break
    }
  }
  r := SparseConstFloat64VectorIterator{k, obj}
  return &r
}
func (obj SparseConstFloat64Vector) JOINT_ITERATOR(b ConstVector) *SparseConstFloat64VectorJointIterator {
  r := SparseConstFloat64VectorJointIterator{}
  r.it1 = obj.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) String() string {
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
func (obj SparseConstFloat64Vector) Table() string {
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
      buffer.WriteString(fmt.Sprintf(" %d:%v", i, float64(0)))
    }
  }
  return buffer.String()
}
/* json
 * -------------------------------------------------------------------------- */
func (obj SparseConstFloat64Vector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []float64{}
  r := struct{
    Index []int
    Value []float64
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetFloat64())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseConstFloat64VectorIterator struct {
  i int
  v SparseConstFloat64Vector
}
func (obj *SparseConstFloat64VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *SparseConstFloat64VectorIterator) GET() ConstFloat64 {
  return ConstFloat64(obj.v.values[obj.i])
}
func (obj *SparseConstFloat64VectorIterator) Ok() bool {
  return obj.i < len(obj.v.indices)
}
func (obj *SparseConstFloat64VectorIterator) Index() int {
  return obj.v.indices[obj.i]
}
func (obj *SparseConstFloat64VectorIterator) Next() {
  obj.i += 1
}
func (obj *SparseConstFloat64VectorIterator) Clone() *SparseConstFloat64VectorIterator {
  return &SparseConstFloat64VectorIterator{obj.i, obj.v}
}
func (obj *SparseConstFloat64VectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseConstFloat64VectorIterator{obj.i, obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseConstFloat64VectorJointIterator struct {
  it1 *SparseConstFloat64VectorIterator
  it2 VectorConstIterator
  idx int
  s1 ConstFloat64
  s2 ConstScalar
}
func (obj *SparseConstFloat64VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseConstFloat64VectorJointIterator) Ok() bool {
  return !(obj.s1.GetFloat64() == float64(0)) ||
         !(obj.s2.GetFloat64() == float64(0))
}
func (obj *SparseConstFloat64VectorJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = ConstFloat64(0)
  obj.s2 = ConstFloat64(0)
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1 = ConstFloat64(0)
      obj.s2 = obj.it2.GetConst()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1 != ConstFloat64(0) {
    obj.it1.Next()
  }
  if obj.s2 != ConstFloat64(0) {
    obj.it2.Next()
  } else {
    obj.s2 = ConstFloat64(0.0)
  }
}
func (obj *SparseConstFloat64VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseConstFloat64VectorJointIterator) GET() (ConstFloat64, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseConstFloat64VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  r := SparseConstFloat64VectorJointIterator{}
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
func (a SparseConstFloat64Vector) Equals(b ConstVector, epsilon float64) bool {
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
