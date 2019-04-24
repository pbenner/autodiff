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

/* vector type declaration
 * -------------------------------------------------------------------------- */

type SparseConstRealVector struct {
  vectorSparseIndexSlice
  values  map[int]ConstReal
  n       int
}

/* constructors
 * -------------------------------------------------------------------------- */

// Allocate a new vector. Scalars are set to the given values.
func NewSparseConstRealVector(indices []int, values []float64, n int) SparseConstRealVector {
  r := NilSparseConstRealVector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if values[i] != 0.0 {
      r.values[k] = ConstReal(values[i])
      r.indexInsert(k)
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

/* -------------------------------------------------------------------------- */

// Create a deep copy of the vector.
func (obj SparseConstRealVector) Clone() SparseConstRealVector {
  r := NilSparseConstRealVector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v
  }
  r.vectorSparseIndexSlice = obj.indexClone()
  return r
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
  for i_k := obj.indexFind(i); obj.index[i_k] < j; i_k++ {
    k := obj.index[i_k]
    r.values[k-i]    = obj.values[k]
    r.indexInsert(k-i)
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

func (obj SparseConstRealVector) ConstRange() chan VectorConstRangeType {
  channel := make(chan VectorConstRangeType)
  go func() {
    obj.indexSort()
    for _, i := range obj.index {
      channel <- VectorConstRangeType{i, obj.values[i]}
    }
    close(channel)
  }()
  return channel
}

func (obj SparseConstRealVector) ConstIterator() VectorConstIterator {
  r := SparseConstRealVectorIterator{obj, -1}
  r.Next()
  return &r
}

func (obj SparseConstRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  r := SparseConstRealVectorJointIterator{}
  r.it1 = &SparseConstRealVectorIterator{obj, -1}
  r.it1.Next()
  r.it2 = b.ConstIterator()
  r.idx = -1
  return &r
}

func (obj SparseConstRealVector) ElementType() ScalarType {
  return BareRealType
}

/* range methods
 * -------------------------------------------------------------------------- */

type SparseConstRealVectorRange struct {
  Index int
  Value ConstReal
}

func (obj SparseConstRealVector) RANGE() chan SparseConstRealVectorRange {
  channel := make(chan SparseConstRealVectorRange)
  go func() {
    obj.indexSort()
    for _, i := range obj.index {
      channel <- SparseConstRealVectorRange{i, obj.values[i]}
    }
    close(channel)
  }()
  return channel
}

type SparseConstRealVectorJointRange struct {
  Index  int
  Value1 ConstReal
  Value2 ConstReal
}

func (obj SparseConstRealVector) JOINT_RANGE(b ConstVector) chan SparseConstRealVectorJointRange {
  channel := make(chan SparseConstRealVectorJointRange)
  go func() {
    c1 := obj.     RANGE()
    c2 :=   b.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    for ok1 || ok2 {
      r := SparseConstRealVectorJointRange{}
      if ok1 {
        r.Index  = r1.Index
        r.Value1 = r1.Value
      }
      if ok2 {
        switch {
        case r.Index >  r2.Index || !ok1:
          r.Index  = r2.Index
          r.Value1 = 0.0
          r.Value2 = ConstReal(r2.Value.GetValue())
        case r.Index == r2.Index:
          r.Value2 = ConstReal(r2.Value.GetValue())
        }
      }
      if r.Value1 != 0.0 {
        r1, ok1 = <- c1
      }
      if r.Value2 != 0.0 {
        r2, ok2 = <- c2
      }
      channel <- r
    }
    close(channel)
  }()
  return channel
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
  for entry := range obj.RANGE() {
    if !first {
      buffer.WriteString(", ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", entry.Index, entry.Value))
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (obj SparseConstRealVector) Table() string {
  var buffer bytes.Buffer

  first := true
  for entry := range obj.RANGE() {
    if !first {
      buffer.WriteString(" ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", entry.Index, entry.Value))
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
  SparseConstRealVector
  i int
}

func (obj *SparseConstRealVectorIterator) GetConst() ConstScalar {
  return ConstReal(obj.values[obj.Index()])
}

func (obj *SparseConstRealVectorIterator) GET() ConstReal {
  return ConstReal(obj.values[obj.Index()])
}

func (obj *SparseConstRealVectorIterator) Ok() bool {
  return obj.i < len(obj.index)
}

func (obj *SparseConstRealVectorIterator) Next() {
  obj.i++
}

func (obj *SparseConstRealVectorIterator) Index() int {
  return obj.index[obj.i]
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
  return obj.it1.Ok() || obj.it2.Ok()
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
