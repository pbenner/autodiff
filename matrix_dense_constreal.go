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

import "fmt"
import "bytes"
import "unsafe"

/* -------------------------------------------------------------------------- */

type DenseConstRealMatrix struct {
	values     []float64
	rows       int
	cols       int
	rowOffset  int
	rowMax     int
	colOffset  int
	colMax     int
	transposed bool
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewDenseConstRealMatrix(rows, cols int, values []float64) DenseConstRealMatrix {
	m := DenseConstRealMatrix{}
	m.values = values
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	return m
}

func NullDenseConstRealMatrix(rows, cols int) DenseConstRealMatrix {
	m := DenseConstRealMatrix{}
	m.values = make([]float64, rows*cols)
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	return m
}

/* -------------------------------------------------------------------------- */

func (matrix DenseConstRealMatrix) CloneConstMatrix() ConstMatrix {
	return matrix.Clone()
}

func (matrix DenseConstRealMatrix) Clone() DenseConstRealMatrix {
	r := matrix
	r.values = make([]float64, len(matrix.values))
	copy(r.values, matrix.values)
	return r
}

/* -------------------------------------------------------------------------- */

func (matrix DenseConstRealMatrix) index(i, j int) int {
	if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
		panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
	}
	if matrix.transposed {
		return (matrix.colOffset+j)*matrix.rowMax + (matrix.rowOffset + i)
	} else {
		return (matrix.rowOffset+i)*matrix.colMax + (matrix.colOffset + j)
	}
}

func (matrix DenseConstRealMatrix) ij(k int) (int, int) {
	if matrix.transposed {
		i := (k % matrix.colMax) - matrix.colOffset
		j := (k / matrix.colMax) - matrix.rowOffset
		return i, j
	} else {
		i := (k % matrix.rowMax) - matrix.rowOffset
		j := (k / matrix.rowMax) - matrix.colOffset
		return i, j
	}
}

func (matrix DenseConstRealMatrix) storageLocation() uintptr {
	return uintptr(unsafe.Pointer(&matrix.values[0]))
}

func (matrix DenseConstRealMatrix) ElementType() ScalarType {
	return BareRealType
}

func (matrix DenseConstRealMatrix) Dims() (int, int) {
	return matrix.rows, matrix.cols
}

func (matrix DenseConstRealMatrix) ValueAt(i, j int) float64 {
	return matrix.values[matrix.index(i, j)]
}

func (matrix DenseConstRealMatrix) ConstAt(i, j int) ConstScalar {
	return ConstReal(matrix.values[matrix.index(i, j)])
}

func (matrix DenseConstRealMatrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
	m := matrix
	m.rowOffset += rfrom
	m.rows = rto - rfrom
	m.colOffset += cfrom
	m.cols = cto - cfrom
	return m
}

func (matrix DenseConstRealMatrix) ConstRow(i int) ConstVector {
	return matrix.ROW(i)
}

func (matrix DenseConstRealMatrix) ROW(i int) DenseConstRealVector {
	var v []float64
	if matrix.transposed {
		v = make([]float64, matrix.cols)
		for j := 0; j < matrix.cols; j++ {
			v[j] = matrix.values[matrix.index(i, j)]
		}
	} else {
		i = matrix.index(i, 0)
		v = matrix.values[i : i+matrix.cols]
	}
	return DenseConstRealVector(v)
}

func (matrix DenseConstRealMatrix) ConstCol(i int) ConstVector {
	return matrix.COL(i)
}

func (matrix DenseConstRealMatrix) COL(j int) DenseConstRealVector {
	var v []float64
	if matrix.transposed {
		j = matrix.index(0, j)
		v = matrix.values[j : j+matrix.rows]
	} else {
		v = make([]float64, matrix.rows)
		for i := 0; i < matrix.rows; i++ {
			v[i] = matrix.values[matrix.index(i, j)]
		}
	}
	return v
}

func (matrix DenseConstRealMatrix) ConstDiag() ConstVector {
	return matrix.DIAG()
}

func (matrix DenseConstRealMatrix) DIAG() DenseConstRealVector {
	n, m := matrix.Dims()
	if n != m {
		panic("Diag(): not a square matrix!")
	}
	v := make([]float64, n)
	for i := 0; i < n; i++ {
		v[i] = matrix.values[matrix.index(i, i)]
	}
	return DenseConstRealVector(v)
}

func (matrix DenseConstRealMatrix) GetValues() []float64 {
	return matrix.values
}

func (matrix DenseConstRealMatrix) AsConstVector() ConstVector {
	return DenseConstRealVector(matrix.values)
}

func (matrix DenseConstRealMatrix) IsSymmetric(epsilon float64) bool {
	n, m := matrix.Dims()
	if n != m {
		return false
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < m; j++ {
			if !matrix.ConstAt(i, j).Equals(matrix.ConstAt(j, i), 1e-12) {
				return false
			}
		}
	}
	return true
}

/* -------------------------------------------------------------------------- */

func (m DenseConstRealMatrix) String() string {
	var buffer bytes.Buffer
	buffer.WriteString("[")
	for i := 0; i < m.rows; i++ {
		if i != 0 {
			buffer.WriteString(",\n ")
		}
		buffer.WriteString("[")
		for j := 0; j < m.cols; j++ {
			if j != 0 {
				buffer.WriteString(", ")
			}
			buffer.WriteString(m.ConstAt(i, j).String())
		}
		buffer.WriteString("]")
	}
	buffer.WriteString("]")
	return buffer.String()
}

func (a DenseConstRealMatrix) Table() string {
	var buffer bytes.Buffer
	n, m := a.Dims()
	for i := 0; i < n; i++ {
		if i != 0 {
			buffer.WriteString("\n")
		}
		for j := 0; j < m; j++ {
			if j != 0 {
				buffer.WriteString(" ")
			}
			buffer.WriteString(a.ConstAt(i, j).String())
		}
	}
	return buffer.String()
}

/* implement ConstScalarContainer
 * -------------------------------------------------------------------------- */

func (matrix DenseConstRealMatrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r = f(r, matrix.ConstAt(i, j))
		}
	}
	return r
}

/* math
 * -------------------------------------------------------------------------- */

func (a DenseConstRealMatrix) Equals(b ConstMatrix, epsilon float64) bool {
	n1, m1 := a.Dims()
	n2, m2 := b.Dims()
	if n1 != n2 || m1 != m2 {
		panic("MEqual(): matrix dimensions do not match!")
	}
	for i := 0; i < n1; i++ {
		for j := 0; j < m1; j++ {
			if !a.ConstAt(i, j).Equals(b.ConstAt(i, j), epsilon) {
				return false
			}
		}
	}
	return true
}

/* iterator methods
 * -------------------------------------------------------------------------- */

func (m DenseConstRealMatrix) ConstIterator() MatrixConstIterator {
	return m.ITERATOR()
}

func (m DenseConstRealMatrix) ITERATOR() *DenseConstRealMatrixIterator {
	r := DenseConstRealMatrixIterator{m, -1}
	r.Next()
	return &r
}

/* const iterator
 * -------------------------------------------------------------------------- */

type DenseConstRealMatrixIterator struct {
	m DenseConstRealMatrix
	i int
}

func (obj *DenseConstRealMatrixIterator) GetConst() ConstScalar {
	return ConstReal(obj.m.values[obj.i])
}

func (obj *DenseConstRealMatrixIterator) GetValue() float64 {
	return obj.m.values[obj.i]
}

func (obj *DenseConstRealMatrixIterator) GET() ConstReal {
	return ConstReal(obj.m.values[obj.i])
}

func (obj *DenseConstRealMatrixIterator) Ok() bool {
	return obj.i < len(obj.m.values)
}

func (obj *DenseConstRealMatrixIterator) Next() {
	obj.i++
}

func (obj *DenseConstRealMatrixIterator) Index() (int, int) {
	return obj.m.ij(obj.i)
}

func (obj *DenseConstRealMatrixIterator) Clone() *DenseConstRealMatrixIterator {
	return &DenseConstRealMatrixIterator{obj.m, obj.i}
}

func (obj *DenseConstRealMatrixIterator) CloneConstIterator() MatrixConstIterator {
	return &DenseConstRealMatrixIterator{obj.m, obj.i}
}
