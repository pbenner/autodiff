/* Copyright (C) 2019 Philipp Benner
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

/* vector type declaration
 * -------------------------------------------------------------------------- */

type DenseGradient struct {
	S Scalar
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) Dim() int {
	return obj.S.GetN()
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) Clone() DenseGradient {
	return DenseGradient{obj.S.CloneScalar()}
}

func (obj DenseGradient) CloneConstVector() ConstVector {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) ValueAt(i int) float64 {
	return obj.S.GetDerivative(i)
}

func (obj DenseGradient) ConstAt(i int) ConstScalar {
	return ConstReal(obj.S.GetDerivative(i))
}

func (obj DenseGradient) AT(i int) ConstReal {
	return ConstReal(obj.S.GetDerivative(i))
}

func (obj DenseGradient) ConstSlice(i, j int) ConstVector {
	x := make([]float64, j-i)
	for k := i; k < j; k++ {
		x[k] = obj.S.GetDerivative(k)
	}
	return NewDenseConstRealVector(x)
}

func (obj DenseGradient) GetValues() []float64 {
	x := make([]float64, obj.Dim())
	for i := 0; i < obj.Dim(); i++ {
		x[i] = obj.S.GetDerivative(i)
	}
	return x
}

func (obj DenseGradient) ConstIterator() VectorConstIterator {
	return obj.ITERATOR()
}

func (obj DenseGradient) ConstJointIterator(b ConstVector) VectorConstJointIterator {
	return obj.JOINT_ITERATOR(b)
}

func (obj DenseGradient) ITERATOR() *DenseGradientIterator {
	r := DenseGradientIterator{obj, -1}
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

func (obj DenseGradient) ElementType() ScalarType {
	return BareRealType
}

/* -------------------------------------------------------------------------- */

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

/* const iterator
 * -------------------------------------------------------------------------- */

type DenseGradientIterator struct {
	v DenseGradient
	i int
}

func (obj *DenseGradientIterator) GetConst() ConstScalar {
	return obj.v.ConstAt(obj.i)
}

func (obj *DenseGradientIterator) GetValue() float64 {
	return obj.v.ValueAt(obj.i)
}

func (obj *DenseGradientIterator) GET() ConstReal {
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
	it2 VectorConstIterator
	idx int
	s1  ConstReal
	s2  ConstReal
}

func (obj *DenseGradientJointIterator) Index() int {
	return obj.idx
}

func (obj *DenseGradientJointIterator) Ok() bool {
	return obj.s1.GetValue() != 0.0 || obj.s2.GetValue() != 0.0
}

func (obj *DenseGradientJointIterator) Next() {
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

func (obj *DenseGradientJointIterator) GetConst() (ConstScalar, ConstScalar) {
	return obj.s1, obj.s2
}

func (obj *DenseGradientJointIterator) GetValue() (float64, float64) {
	return obj.s1.GetValue(), obj.s2.GetValue()
}

func (obj *DenseGradientJointIterator) GET() (ConstReal, ConstScalar) {
	return obj.s1, obj.s2
}

func (obj *DenseGradientJointIterator) CloneConstJointIterator() VectorConstJointIterator {
	r := DenseGradientJointIterator{}
	r.it1 = obj.it1.Clone()
	r.it2 = obj.it2.CloneConstIterator()
	r.idx = obj.idx
	r.s1 = obj.s1
	r.s2 = obj.s2
	return &r
}

/* imlement ConstScalarContainer
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	for i := 0; i < obj.Dim(); i++ {
		r = f(r, obj.ConstAt(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */

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
