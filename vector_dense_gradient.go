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

//import "fmt"
import "bytes"
import "encoding/json"

/* vector type declaration
 * -------------------------------------------------------------------------- */

type DenseGradient struct {
  S Scalar
}

/* cloning
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) Clone() DenseGradient {
  return DenseGradient{obj.S.CloneScalar()}
}

/* native vector methods
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) AT(i int) ConstFloat64 {
  return ConstFloat64(obj.S.GetDerivative(i))
}

/* const interface
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) CloneConstVector() ConstVector {
  return obj.Clone()
}

func (obj DenseGradient) Dim() int {
  return obj.S.GetN()
}

func (obj DenseGradient) Int8At(i int) int8 {
  return int8(obj.S.GetDerivative(i))
}

func (obj DenseGradient) Int16At(i int) int16 {
  return int16(obj.S.GetDerivative(i))
}

func (obj DenseGradient) Int32At(i int) int32 {
  return int32(obj.S.GetDerivative(i))
}

func (obj DenseGradient) Int64At(i int) int64 {
  return int64(obj.S.GetDerivative(i))
}

func (obj DenseGradient) IntAt(i int) int {
  return int(obj.S.GetDerivative(i))
}

func (obj DenseGradient) Float32At(i int) float32 {
  return float32(obj.S.GetDerivative(i))
}

func (obj DenseGradient) Float64At(i int) float64 {
  return float64(obj.S.GetDerivative(i))
}

func (obj DenseGradient) ConstAt(i int) ConstScalar {
  return ConstFloat64(obj.S.GetDerivative(i))
}

func (obj DenseGradient) ConstSlice(i, j int) ConstVector {
  x := make([]float64, j-i)
  for k := i; k < j; k++ {
    x[k] = obj.S.GetDerivative(k)
  }
  return NewDenseFloat64Vector(x)
}

func (obj DenseGradient) AsConstMatrix(n, m int) ConstMatrix {
  if n*m != obj.Dim() {
    panic("Matrix dimension does not fit input vector!")
  }
  r := NullDenseFloat64Vector(obj.Dim())
  r.Set(obj)
  return r.AsConstMatrix(n, m)
}

/* imlement ConstScalarContainer
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < obj.Dim(); i++ {
    r = f(r, obj.ConstAt(i))
  }
  return r
}

func (obj DenseGradient) ElementType() ScalarType {
  return ConstFloat64Type
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i := 0; i < obj.Dim(); i++ {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(obj.ConstAt(i).String())
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (obj DenseGradient) Table() string {
  var buffer bytes.Buffer

  for i := 0; i < obj.Dim(); i++ {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(obj.ConstAt(i).String())
  }

  return buffer.String()
}

/* json
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) MarshalJSON() ([]byte, error) {
  r := make([]float64, obj.Dim())
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    r[it.Index()] = it.GetConst().GetFloat64()
  }
  return json.MarshalIndent(r, "", "  ")
}

/* iterator methods
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}

func (obj DenseGradient) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}

func (obj DenseGradient) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}

func (obj DenseGradient) ITERATOR() *DenseGradientIterator {
  r := DenseGradientIterator{obj, -1}
  r.Next()
  return &r
}

func (obj DenseGradient) ITERATOR_FROM(i int) *DenseGradientIterator {
  r := DenseGradientIterator{obj, i-1}
  r.Next()
  return &r
}

func (obj DenseGradient) JOINT_ITERATOR(b ConstVector) *DenseGradientJointIterator {
  r := DenseGradientJointIterator{}
  r.it1 = obj.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}

/* const iterator
 * -------------------------------------------------------------------------- */

type DenseGradientIterator struct {
  v DenseGradient
  i int
}

func (obj *DenseGradientIterator) GetConst() ConstScalar {
  return obj.v.ConstAt(obj.i)
}

func (obj *DenseGradientIterator) GET() ConstFloat64 {
  return obj.v.AT(obj.i)
}

func (obj *DenseGradientIterator) Ok() bool {
  return obj.i < obj.v.Dim()
}

func (obj *DenseGradientIterator) Next() {
  obj.i++
}

func (obj *DenseGradientIterator) Index() int {
  return obj.i
}

func (obj *DenseGradientIterator) Clone() *DenseGradientIterator {
  return &DenseGradientIterator{obj.v, obj.i}
}

func (obj *DenseGradientIterator) CloneConstIterator() VectorConstIterator {
  return &DenseGradientIterator{obj.v, obj.i}
}

/* joint iterator
 * -------------------------------------------------------------------------- */

type DenseGradientJointIterator struct {
  it1 *DenseGradientIterator
  it2  VectorConstIterator
  idx  int
  s1   ConstFloat64
  s2   ConstScalar
}

func (obj *DenseGradientJointIterator) Index() int {
  return obj.idx
}

func (obj *DenseGradientJointIterator) Ok() bool {
  return obj.s1.GetFloat64() != 0.0 || !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}

func (obj *DenseGradientJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = 0.0
  obj.s2 = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1  = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx >  obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1  = 0.0
      obj.s2  = obj.it2.GetConst()
    case obj.idx == obj.it2.Index():
      obj.s2  = obj.it2.GetConst()
    }
  }
}

func (obj *DenseGradientJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}

func (obj *DenseGradientJointIterator) GET() (ConstFloat64, ConstScalar) {
  return obj.s1, obj.s2
}

func (obj *DenseGradientJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  r := DenseGradientJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1  = obj.s1
  r.s2  = obj.s2
  return &r
}

/* math
 * -------------------------------------------------------------------------- */

// Test if elements in a equal elements in b.
func (a DenseGradient) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for i := 0; i < a.Dim(); i++ {
    if !a.ConstAt(i).Equals(b.ConstAt(i), epsilon) {
      return false
    }
  }
  return true
}
