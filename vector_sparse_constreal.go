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

import "fmt"
import "bytes"
import "sort"

/* vector type declaration
 * -------------------------------------------------------------------------- */

type SparseConstRealVector struct {
  values  map[int]ConstReal
  indices []int
  n       int
}

/* constructors
 * -------------------------------------------------------------------------- */

// Allocate a new vector. Scalars are set to the given values.
func NewSparseConstRealVector(indices []int, values []float64, n int) SparseConstRealVector {
  if len(indices) != len(values) {
    panic("invalid number of indices")
  }
  sort.Sort(sortIntFloat{indices, values})
  r := NilSparseConstRealVector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if values[i] != 0.0 {
      r.values[k] = ConstReal(values[i])
      r.indices   = append(r.indices, indices[i])
    }
  }
  return r
}

func NilSparseConstRealVector(n int) SparseConstRealVector {
  r := SparseConstRealVector{}
  r.n      = n
  r.values = make(map[int]ConstReal)
  return r
}

// Convert vector type.
func AsSparseConstRealVector(v ConstVector) SparseConstRealVector {
  switch v_ := v.(type) {
  case SparseConstRealVector:
    return v_
  }
  indices := []int{}
  values  := []float64{}
  n       := v.Dim()
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    indices = append(indices, it.Index())
    values  = append(values,  it.GetConst().GetValue())
  }
  return NewSparseConstRealVector(indices, values, n)
}

/* const vector methods
 * -------------------------------------------------------------------------- */

func (obj SparseConstRealVector) Dim() int {
  return obj.n
}

func (obj SparseConstRealVector) ValueAt(i int) float64 {
  if v, ok := obj.values[i]; ok {
    return v.GetValue()
  } else {
    return 0.0
  }
}

func (obj SparseConstRealVector) ConstAt(i int) ConstScalar {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstReal(0.0)
  }
}

func (obj SparseConstRealVector) ConstSlice(i, j int) ConstVector {
  r := NilSparseConstRealVector(j-i)
  for _, k := range obj.indices {
    if k < i {
      continue
    }
    if k >= j {
      break
    }
    r.values[k-i] = obj.values[k]
    r.indices     = append(r.indices, k-i)
  }
  return r
}

func (obj SparseConstRealVector) GetValues() []float64 {
  r := make([]float64, obj.Dim())
  for i, v := range obj.values {
    r[i] = v.GetValue()
  }
  return r
}

func (obj SparseConstRealVector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}

func (obj SparseConstRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}

func (obj SparseConstRealVector) ITERATOR() *SparseConstRealVectorIterator {
  r := SparseConstRealVectorIterator{0, obj}
  return &r
}

func (obj SparseConstRealVector) JOINT_ITERATOR(b ConstVector) *SparseConstRealVectorJointIterator {
  r := SparseConstRealVectorJointIterator{}
  r.it1 = obj.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}

func (obj SparseConstRealVector) ElementType() ScalarType {
  return BareRealType
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */

func (obj SparseConstRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (obj SparseConstRealVector) String() string {
  var buffer bytes.Buffer

  buffer.WriteString(fmt.Sprintf("%d:[", obj.n))
  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(", ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", it.Index(), it.GetConst()))
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (obj SparseConstRealVector) Table() string {
  var buffer bytes.Buffer

  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(" ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", it.Index(), it.GetConst()))
  }
  if _, ok := obj.values[obj.n-1]; !ok {
    i := obj.n-1
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", i, ConstReal(0.0)))
  }

  return buffer.String()
}

/* -------------------------------------------------------------------------- */

// Test if elements in a equal elements in b.
func (a SparseConstRealVector) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for it := a.ConstJointIterator(b); it.Ok(); it.Next() {
    s1, s2 := it.GetConst()
    if !s1.Equals(s2, epsilon) {
      return false
    }
  }
  return true
}

/* const iterator
 * -------------------------------------------------------------------------- */

type SparseConstRealVectorIterator struct {
  i int
  v SparseConstRealVector
}

func (obj *SparseConstRealVectorIterator) GetConst() ConstScalar {
  return obj.GET()
}

func (obj *SparseConstRealVectorIterator) GET() ConstReal {
  return obj.v.values[obj.Index()]
}

func (obj *SparseConstRealVectorIterator) Ok() bool {
  return obj.i < len(obj.v.indices)
}

func (obj *SparseConstRealVectorIterator) Index() int {
  return obj.v.indices[obj.i]
}

func (obj *SparseConstRealVectorIterator) Next() {
  obj.i += 1
}

/* joint iterator
 * -------------------------------------------------------------------------- */

type SparseConstRealVectorJointIterator struct {
  it1 *SparseConstRealVectorIterator
  it2  VectorConstIterator
  idx  int
  s1   ConstReal
  s2   ConstReal
}

func (obj *SparseConstRealVectorJointIterator) Index() int {
  return obj.idx
}

func (obj *SparseConstRealVectorJointIterator) Ok() bool {
  return obj.s1.GetValue() != 0.0 || obj.s2.GetValue() != 0.0
}

func (obj *SparseConstRealVectorJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = 0.0
  obj.s2 = 0.0
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1  = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx >  obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1  = 0.0
      obj.s2  = ConstReal(obj.it2.GetConst().GetValue())
    case obj.idx == obj.it2.Index():
      obj.s2  = ConstReal(obj.it2.GetConst().GetValue())
    }
  }
  if obj.s1 != 0.0 {
    obj.it1.Next()
  }
  if obj.s2 != 0.0 {
    obj.it2.Next()
  } else {
    obj.s2 = 0.0
  }
}

func (obj *SparseConstRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}

func (obj *SparseConstRealVectorJointIterator) GET() (ConstReal, ConstScalar) {
  return obj.s1, obj.s2
}
