/* Copyright (C) 2015-2018 Philipp Benner
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

/* -------------------------------------------------------------------------- */

type DenseConstRealVector []float64

/* constructors
 * -------------------------------------------------------------------------- */

func NewDenseConstRealVector(v []float64) DenseConstRealVector {
	return DenseConstRealVector(v)
}

func NullDenseConstRealVector(n int) DenseConstRealVector {
	return DenseConstRealVector(make([]float64, n))
}

// Convert vector type.
func AsDenseConstRealVector(v ConstVector) DenseConstRealVector {
	switch v_ := v.(type) {
	case DenseConstRealVector:
		return v_
	}
	values := make([]float64, v.Dim())
	for it := v.ConstIterator(); it.Ok(); it.Next() {
		values[it.Index()] = it.GetValue()
	}
	return NewDenseConstRealVector(values)
}

/* -------------------------------------------------------------------------- */

func (v DenseConstRealVector) Clone() DenseConstRealVector {
	r := make([]float64, v.Dim())
	copy(r, v)
	return r
}

/* -------------------------------------------------------------------------- */

func (v DenseConstRealVector) Dim() int {
	return len(v)
}

func (v DenseConstRealVector) ValueAt(i int) float64 {
	return v[i]
}

func (v DenseConstRealVector) ConstAt(i int) ConstScalar {
	return ConstReal(v[i])
}

func (v DenseConstRealVector) ConstSlice(i, j int) ConstVector {
	return v[i:j]
}

func (v DenseConstRealVector) GetValues() []float64 {
	return v
}

func (v DenseConstRealVector) ConstIterator() VectorConstIterator {
	return v.ITERATOR()
}

func (v DenseConstRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
	return v.JOINT_ITERATOR(b)
}

func (v DenseConstRealVector) ITERATOR() *DenseConstRealVectorIterator {
	r := DenseConstRealVectorIterator{v, -1}
	r.Next()
	return &r
}

func (v DenseConstRealVector) JOINT_ITERATOR(b ConstVector) *DenseConstRealVectorJointIterator {
	r := DenseConstRealVectorJointIterator{}
	r.it1 = v.ITERATOR()
	r.it2 = b.ConstIterator()
	r.idx = -1
	r.Next()
	return &r
}

func (v DenseConstRealVector) ElementType() ScalarType {
	return BareRealType
}

/* -------------------------------------------------------------------------- */

func (v DenseConstRealVector) String() string {
	var buffer bytes.Buffer
	buffer.WriteString("[")
	for i, _ := range v {
		if i != 0 {
			buffer.WriteString(", ")
		}
		buffer.WriteString(v.ConstAt(i).String())
	}
	buffer.WriteString("]")
	return buffer.String()
}

func (v DenseConstRealVector) Table() string {
	var buffer bytes.Buffer
	for i, _ := range v {
		if i != 0 {
			buffer.WriteString(" ")
		}
		buffer.WriteString(v.ConstAt(i).String())
	}
	return buffer.String()
}

/* imlement ConstScalarContainer
 * -------------------------------------------------------------------------- */

func (v DenseConstRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	for i := 0; i < len(v); i++ {
		r = f(r, v.ConstAt(i))
	}
	return r
}

/* math
 * -------------------------------------------------------------------------- */

func (a DenseConstRealVector) Equals(b ConstVector, epsilon float64) bool {
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

/* const iterator
 * -------------------------------------------------------------------------- */

type DenseConstRealVectorIterator struct {
	v DenseConstRealVector
	i int
}

func (obj *DenseConstRealVectorIterator) GetConst() ConstScalar {
	return ConstReal(obj.v[obj.i])
}

func (obj *DenseConstRealVectorIterator) GetValue() float64 {
	return obj.v[obj.i]
}

func (obj *DenseConstRealVectorIterator) GET() ConstReal {
	return ConstReal(obj.v[obj.i])
}

func (obj *DenseConstRealVectorIterator) Ok() bool {
	return obj.i < len(obj.v)
}

func (obj *DenseConstRealVectorIterator) Next() {
	obj.i++
}

func (obj *DenseConstRealVectorIterator) Index() int {
	return obj.i
}

func (obj *DenseConstRealVectorIterator) Clone() *DenseConstRealVectorIterator {
	return &DenseConstRealVectorIterator{obj.v, obj.i}
}

func (obj *DenseConstRealVectorIterator) CloneConstIterator() VectorConstIterator {
	return &DenseConstRealVectorIterator{obj.v, obj.i}
}

/* joint iterator
 * -------------------------------------------------------------------------- */

type DenseConstRealVectorJointIterator struct {
	it1 *DenseConstRealVectorIterator
	it2 VectorConstIterator
	idx int
	s1  ConstReal
	s2  ConstReal
}

func (obj *DenseConstRealVectorJointIterator) Index() int {
	return obj.idx
}

func (obj *DenseConstRealVectorJointIterator) Ok() bool {
	return obj.s1.GetValue() != 0.0 || obj.s2.GetValue() != 0.0
}

func (obj *DenseConstRealVectorJointIterator) Next() {
	ok1 := obj.it1.Ok()
	ok2 := obj.it2.Ok()
	obj.s1 = 0.0
	obj.s2 = 0.0
	if ok1 {
		obj.idx = obj.it1.Index()
		obj.s1 = obj.it1.GET()
	}
	if ok2 {
		switch {
		case obj.idx > obj.it2.Index() || !ok1:
			obj.idx = obj.it2.Index()
			obj.s1 = 0.0
			obj.s2 = ConstReal(obj.it2.GetValue())
		case obj.idx == obj.it2.Index():
			obj.s2 = ConstReal(obj.it2.GetValue())
		}
	}
}

func (obj *DenseConstRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
	return obj.s1, obj.s2
}

func (obj *DenseConstRealVectorJointIterator) GetValue() (float64, float64) {
	return obj.s1.GetValue(), obj.s2.GetValue()
}

func (obj *DenseConstRealVectorJointIterator) GET() (ConstReal, ConstScalar) {
	return obj.s1, obj.s2
}

func (obj *DenseConstRealVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
	r := DenseConstRealVectorJointIterator{}
	r.it1 = obj.it1.Clone()
	r.it2 = obj.it2.CloneConstIterator()
	r.idx = obj.idx
	r.s1 = obj.s1
	r.s2 = obj.s2
	return &r
}
