/* Copyright (C) 2017 Philipp Benner
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

import "encoding/json"

/* -------------------------------------------------------------------------- */

type VectorConstIterator interface {
	CloneConstIterator() VectorConstIterator
	GetValue() float64
	GetConst() ConstScalar
	Ok() bool
	Next()
	Index() int
}

type VectorConstJointIterator interface {
	CloneConstJointIterator() VectorConstJointIterator
	GetValue() (float64, float64)
	GetConst() (ConstScalar, ConstScalar)
	Ok() bool
	Next()
	Index() int
}

type VectorIterator interface {
	CloneIterator() VectorIterator
	GetConst() ConstScalar
	GetValue() float64
	Get() Scalar
	Ok() bool
	Next()
	Index() int
}

type VectorJointIterator interface {
	CloneJointIterator() VectorJointIterator
	GetConst() (ConstScalar, ConstScalar)
	GetValue() (float64, float64)
	Get() (Scalar, ConstScalar)
	Ok() bool
	Next()
	Index() int
}

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type ConstVector interface {
	ConstScalarContainer
	Dim() int
	Equals(ConstVector, float64) bool
	Table() string
	ValueAt(int) float64
	ConstAt(int) ConstScalar
	ConstSlice(i, j int) ConstVector
	GetValues() []float64
	ConstIterator() VectorConstIterator
	ConstJointIterator(ConstVector) VectorConstJointIterator
}

type Vector interface {
	ScalarContainer
	// const methods
	Dim() int
	Equals(ConstVector, float64) bool
	Table() string
	ValueAt(int) float64
	ConstAt(int) ConstScalar
	ConstSlice(i, j int) ConstVector
	GetValues() []float64
	// iterators
	ConstIterator() VectorConstIterator
	JointIterator(ConstVector) VectorJointIterator
	ConstJointIterator(ConstVector) VectorConstJointIterator
	Iterator() VectorIterator
	// other methods
	At(int) Scalar
	Reset()
	ResetDerivatives()
	// basic methods
	CloneVector() Vector
	Set(ConstVector)
	Export(string) error
	Permute([]int) error
	ReverseOrder()
	Sort(bool)
	Slice(i, j int) Vector
	AppendScalar(...Scalar) Vector
	AppendVector(Vector) Vector
	Swap(i, j int)
	// type conversions
	AsMatrix(n, m int) Matrix
	// math operations
	VaddV(a, b ConstVector) Vector
	VaddS(a ConstVector, b ConstScalar) Vector
	VsubV(a, b ConstVector) Vector
	VsubS(a ConstVector, b ConstScalar) Vector
	VmulV(a, b ConstVector) Vector
	VmulS(a ConstVector, b ConstScalar) Vector
	VdivV(a, b ConstVector) Vector
	VdivS(a ConstVector, b ConstScalar) Vector
	MdotV(a ConstMatrix, b ConstVector) Vector
	VdotM(a ConstVector, b ConstMatrix) Vector
	// json
	json.Marshaler
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewVector(t ScalarType, values []float64) Vector {
	switch t {
	case RealType:
		return NewDenseRealVector(values)
	case BareRealType:
		return NewDenseBareRealVector(values)
	default:
		panic("unknown type")
	}
}

func NullVector(t ScalarType, length int) Vector {
	switch t {
	case RealType:
		return NullDenseRealVector(length)
	case BareRealType:
		return NullDenseBareRealVector(length)
	default:
		panic("unknown type")
	}
}

func AsVector(t ScalarType, v Vector) Vector {
	switch t {
	case RealType:
		return AsDenseRealVector(v)
	case BareRealType:
		return AsDenseBareRealVector(v)
	default:
		panic("unknown type")
	}
}
