/* Copyright (C) 2015 Philipp Benner
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

type MatrixConstIterator interface {
	CloneConstIterator() MatrixConstIterator
	GetValue() float64
	GetConst() ConstScalar
	Ok() bool
	Next()
	Index() (int, int)
}

type MatrixConstJointIterator interface {
	CloneConstJointIterator() MatrixConstJointIterator
	GetValue() (float64, float64)
	GetConst() (ConstScalar, ConstScalar)
	Ok() bool
	Next()
	Index() (int, int)
}

type MatrixIterator interface {
	CloneIterator() MatrixIterator
	GetConst() ConstScalar
	GetValue() float64
	Get() Scalar
	Ok() bool
	Next()
	Index() (int, int)
}

type MatrixJointIterator interface {
	CloneJointIterator() MatrixJointIterator
	GetConst() (ConstScalar, ConstScalar)
	GetValue() (float64, float64)
	Get() (Scalar, ConstScalar)
	Ok() bool
	Next()
	Index() (int, int)
}

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type ConstMatrix interface {
	ConstScalarContainer
	Dims() (int, int)
	Equals(ConstMatrix, float64) bool
	Table() string
	ValueAt(i, j int) float64
	ConstAt(i, j int) ConstScalar
	ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix
	ConstRow(i int) ConstVector
	ConstCol(j int) ConstVector
	ConstDiag() ConstVector
	GetValues() []float64
	AsConstVector() ConstVector
	CloneConstMatrix() ConstMatrix
	ConstIterator() MatrixConstIterator
	IsSymmetric(float64) bool
	// private methods
	storageLocation() uintptr
}

type Matrix interface {
	ScalarContainer
	// const methods
	Dims() (int, int)
	Equals(ConstMatrix, float64) bool
	Table() string
	ValueAt(i, j int) float64
	ConstAt(i, j int) ConstScalar
	ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix
	ConstRow(i int) ConstVector
	ConstCol(j int) ConstVector
	ConstDiag() ConstVector
	GetValues() []float64
	// other methods
	At(i, j int) Scalar
	Reset()
	ResetDerivatives()
	// basic methods
	CloneMatrix() Matrix
	CloneConstMatrix() ConstMatrix
	Set(ConstMatrix)
	Row(i int) Vector
	Col(j int) Vector
	Diag() Vector
	T() Matrix
	Tip()
	Export(string) error
	Slice(rfrom, rto, cfrom, cto int) Matrix
	Swap(int, int, int, int)
	SwapRows(int, int) error
	SwapColumns(int, int) error
	PermuteRows([]int) error
	PermuteColumns([]int) error
	SymmetricPermutation(pi []int) error
	SetIdentity()
	IsSymmetric(float64) bool
	// returns all elements of the matrix as
	// a vector, the order is unspecified
	AsVector() Vector
	AsConstVector() ConstVector
	// iterators
	ConstIterator() MatrixConstIterator
	Iterator() MatrixIterator
	// math operations
	MaddM(a, b ConstMatrix) Matrix
	MaddS(a ConstMatrix, b ConstScalar) Matrix
	MsubM(a, b ConstMatrix) Matrix
	MsubS(a ConstMatrix, b ConstScalar) Matrix
	MmulM(a, b ConstMatrix) Matrix
	MmulS(a ConstMatrix, b ConstScalar) Matrix
	MdivM(a, b ConstMatrix) Matrix
	MdivS(a ConstMatrix, b ConstScalar) Matrix
	MdotM(a, b ConstMatrix) Matrix
	Outer(a, b ConstVector) Matrix
	Jacobian(f func(ConstVector) ConstVector, x_ Vector) Matrix
	Hessian(f func(ConstVector) ConstScalar, x_ Vector) Matrix
	// json
	json.Marshaler
	// private methods
	storageLocation() uintptr
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewMatrix(t ScalarType, rows, cols int, values []float64) Matrix {
	switch t {
	case RealType:
		return NewDenseRealMatrix(rows, cols, values)
	case BareRealType:
		return NewDenseBareRealMatrix(rows, cols, values)
	default:
		panic("unknown type")
	}
}

func NullMatrix(t ScalarType, rows, cols int) Matrix {
	switch t {
	case RealType:
		return NullDenseRealMatrix(rows, cols)
	case BareRealType:
		return NullDenseBareRealMatrix(rows, cols)
	default:
		panic("unknown type")
	}
}

func AsMatrix(t ScalarType, m Matrix) Matrix {
	switch t {
	case RealType:
		return AsDenseRealMatrix(m)
	case BareRealType:
		return AsDenseBareRealMatrix(m)
	default:
		panic("unknown type")
	}
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func IdentityMatrix(t ScalarType, dim int) Matrix {
	matrix := NullMatrix(t, dim, dim)
	for i := 0; i < dim; i++ {
		matrix.At(i, i).SetValue(1.0)
	}
	return matrix
}
