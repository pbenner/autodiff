/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2017 Philipp Benner
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
package autodiff

/* -------------------------------------------------------------------------- */
import "bytes"
import "bufio"
import "compress/gzip"
import "encoding/json"
import "fmt"
import "strconv"
import "strings"
import "os"
import "unsafe"

/* -------------------------------------------------------------------------- */
/* matrix type declaration
 * -------------------------------------------------------------------------- */
type DenseRealMatrix struct {
	values     DenseRealVector
	rows       int
	cols       int
	rowOffset  int
	rowMax     int
	colOffset  int
	colMax     int
	transposed bool
	tmp1       DenseRealVector
	tmp2       DenseRealVector
}

/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseRealMatrix(rows, cols int, values []float64) *DenseRealMatrix {
	m := nilDenseRealMatrix(rows, cols)
	v := m.values
	if len(values) == 1 {
		for i := 0; i < rows*cols; i++ {
			v[i] = NewReal(values[0])
		}
	} else if len(values) == rows*cols {
		for i := 0; i < rows*cols; i++ {
			v[i] = NewReal(values[i])
		}
	} else {
		panic("NewMatrix(): Matrix dimension does not fit input values!")
	}
	m.initTmp()
	return m
}
func NullDenseRealMatrix(rows, cols int) *DenseRealMatrix {
	m := DenseRealMatrix{}
	m.values = NullDenseRealVector(rows * cols)
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	m.initTmp()
	return &m
}
func nilDenseRealMatrix(rows, cols int) *DenseRealMatrix {
	m := DenseRealMatrix{}
	m.values = nilDenseRealVector(rows * cols)
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	return &m
}
func AsDenseRealMatrix(matrix ConstMatrix) *DenseRealMatrix {
	switch matrix_ := matrix.(type) {
	case *DenseRealMatrix:
		return matrix_.Clone()
	}
	n, m := matrix.Dims()
	r := NullDenseRealMatrix(n, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r.AT(i, j).Set(matrix.ConstAt(i, j))
		}
	}
	return r
}
func (matrix *DenseRealMatrix) initTmp() {
	if len(matrix.tmp1) < matrix.rows {
		matrix.tmp1 = NullDenseRealVector(matrix.rows)
	} else {
		matrix.tmp1 = matrix.tmp1[0:matrix.rows]
	}
	if len(matrix.tmp2) < matrix.cols {
		matrix.tmp2 = NullDenseRealVector(matrix.cols)
	} else {
		matrix.tmp2 = matrix.tmp2[0:matrix.cols]
	}
}

/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *DenseRealMatrix) Clone() *DenseRealMatrix {
	return &DenseRealMatrix{
		values:     matrix.values.Clone(),
		rows:       matrix.rows,
		cols:       matrix.cols,
		transposed: matrix.transposed,
		rowOffset:  matrix.rowOffset,
		rowMax:     matrix.rowMax,
		colOffset:  matrix.colOffset,
		colMax:     matrix.colMax,
		tmp1:       matrix.tmp1.Clone(),
		tmp2:       matrix.tmp2.Clone()}
}
func (matrix *DenseRealMatrix) CloneMatrix() Matrix {
	return matrix.Clone()
}
func (matrix *DenseRealMatrix) CloneConstMatrix() ConstMatrix {
	return matrix.Clone()
}

/* field access
 * -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) index(i, j int) int {
	if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
		panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
	}
	if matrix.transposed {
		return (matrix.colOffset+j)*matrix.rowMax + (matrix.rowOffset + i)
	} else {
		return (matrix.rowOffset+i)*matrix.colMax + (matrix.colOffset + j)
	}
}
func (matrix *DenseRealMatrix) ij(k int) (int, int) {
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
func (matrix *DenseRealMatrix) Dims() (int, int) {
	if matrix == nil {
		return 0, 0
	} else {
		return matrix.rows, matrix.cols
	}
}
func (matrix *DenseRealMatrix) Row(i int) Vector {
	return matrix.ROW(i)
}
func (matrix *DenseRealMatrix) ROW(i int) DenseRealVector {
	var v DenseRealVector
	if matrix.transposed {
		v = nilDenseRealVector(matrix.cols)
		for j := 0; j < matrix.cols; j++ {
			v[j] = matrix.values[matrix.index(i, j)]
		}
	} else {
		i = matrix.index(i, 0)
		v = matrix.values[i : i+matrix.cols]
	}
	return v
}
func (matrix *DenseRealMatrix) Col(j int) Vector {
	return matrix.COL(j)
}
func (matrix *DenseRealMatrix) COL(j int) DenseRealVector {
	var v DenseRealVector
	if matrix.transposed {
		j = matrix.index(0, j)
		v = matrix.values[j : j+matrix.rows]
	} else {
		v = nilDenseRealVector(matrix.rows)
		for i := 0; i < matrix.rows; i++ {
			v[i] = matrix.values[matrix.index(i, j)]
		}
	}
	return v
}
func (matrix *DenseRealMatrix) Diag() Vector {
	return matrix.DIAG()
}
func (matrix *DenseRealMatrix) DIAG() DenseRealVector {
	n, m := matrix.Dims()
	if n != m {
		panic("Diag(): not a square matrix!")
	}
	v := nilDenseRealVector(n)
	for i := 0; i < n; i++ {
		v[i] = matrix.values[matrix.index(i, i)]
	}
	return v
}
func (matrix *DenseRealMatrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
	m := *matrix
	m.rowOffset += rfrom
	m.rows = rto - rfrom
	m.colOffset += cfrom
	m.cols = cto - cfrom
	// crop tmp vectors
	m.initTmp()
	return &m
}
func (matrix *DenseRealMatrix) Swap(i1, j1, i2, j2 int) {
	k1 := matrix.index(i1, j1)
	k2 := matrix.index(i2, j2)
	matrix.values[k1], matrix.values[k2] = matrix.values[k2], matrix.values[k1]
}
func (matrix *DenseRealMatrix) AsVector() Vector {
	return matrix.AsDenseRealVector()
}
func (matrix *DenseRealMatrix) AsConstVector() ConstVector {
	return matrix.AsVector()
}
func (matrix *DenseRealMatrix) AsDenseRealVector() DenseRealVector {
	if matrix.cols < matrix.colMax-matrix.colOffset ||
		(matrix.rows < matrix.rowMax-matrix.rowOffset) {
		n, m := matrix.Dims()
		v := nilDenseRealVector(n * m)
		for i := 0; i < n; i++ {
			for j := 0; j < m; j++ {
				v[i*matrix.cols+j] = matrix.AT(i, j)
			}
		}
		return v
	} else {
		return DenseRealVector(matrix.values)
	}
}

/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) T() Matrix {
	return &DenseRealMatrix{
		values:     matrix.values,
		rows:       matrix.cols,
		cols:       matrix.rows,
		transposed: !matrix.transposed,
		rowOffset:  matrix.colOffset,
		rowMax:     matrix.colMax,
		colOffset:  matrix.rowOffset,
		colMax:     matrix.rowMax,
		tmp1:       matrix.tmp2,
		tmp2:       matrix.tmp1}
}
func (matrix *DenseRealMatrix) Tip() {
	mn := len(matrix.values)
	visited := make([]bool, mn)
	k := 0
	for cycle := 1; cycle < mn; cycle++ {
		if visited[cycle] {
			continue
		}
		k = cycle
		for {
			if k != mn-1 {
				k = matrix.rows * k % (mn - 1)
			}
			visited[k] = true
			// swap
			matrix.values[k], matrix.values[cycle] = matrix.values[cycle], matrix.values[k]
			if k == cycle {
				break
			}
		}
	}
	matrix.rows, matrix.cols = matrix.cols, matrix.rows
	matrix.rowOffset, matrix.colOffset = matrix.colOffset, matrix.rowOffset
	matrix.rowMax, matrix.colMax = matrix.colMax, matrix.rowMax
	matrix.tmp1, matrix.tmp2 = matrix.tmp2, matrix.tmp1
}

/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) ValueAt(i, j int) float64 {
	return matrix.values[matrix.index(i, j)].GetValue()
}
func (matrix *DenseRealMatrix) ConstAt(i, j int) ConstScalar {
	return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseRealMatrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
	return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *DenseRealMatrix) ConstRow(i int) ConstVector {
	return matrix.ROW(i)
}
func (matrix *DenseRealMatrix) ConstCol(i int) ConstVector {
	return matrix.COL(i)
}
func (matrix *DenseRealMatrix) ConstDiag() ConstVector {
	return matrix.DIAG()
}
func (matrix *DenseRealMatrix) GetValues() []float64 {
	n, m := matrix.Dims()
	s := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			s[i*m+j] = matrix.ConstAt(i, j).GetValue()
		}
	}
	return s
}

/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) At(i, j int) Scalar {
	return matrix.AT(i, j)
}
func (matrix *DenseRealMatrix) AT(i, j int) *Real {
	return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseRealMatrix) Reset() {
	for i := 0; i < len(matrix.values); i++ {
		matrix.values[i].Reset()
	}
}
func (matrix *DenseRealMatrix) ResetDerivatives() {
	for i := 0; i < len(matrix.values); i++ {
		matrix.values[i].ResetDerivatives()
	}
}
func (a *DenseRealMatrix) Set(b ConstMatrix) {
	n1, m1 := a.Dims()
	n2, m2 := b.Dims()
	if n1 != n2 || m1 != m2 {
		panic("Copy(): Matrix dimension does not match!")
	}
	for i := 0; i < n1; i++ {
		for j := 0; j < m1; j++ {
			a.At(i, j).Set(b.ConstAt(i, j))
		}
	}
}
func (matrix *DenseRealMatrix) SetIdentity() {
	n, m := matrix.Dims()
	c := NewScalar(matrix.ElementType(), 1.0)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if i == j {
				matrix.At(i, j).Set(c)
			} else {
				matrix.At(i, j).Reset()
			}
		}
	}
}
func (matrix *DenseRealMatrix) IsSymmetric(epsilon float64) bool {
	n, m := matrix.Dims()
	if n != m {
		return false
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < m; j++ {
			if !matrix.At(i, j).Equals(matrix.At(j, i), 1e-12) {
				return false
			}
		}
	}
	return true
}
func (matrix *DenseRealMatrix) storageLocation() uintptr {
	return uintptr(unsafe.Pointer(&matrix.values[0]))
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) Map(f func(Scalar)) {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			f(matrix.At(i, j))
		}
	}
}
func (matrix *DenseRealMatrix) MapSet(f func(ConstScalar) Scalar) {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			matrix.At(i, j).Set(f(matrix.ConstAt(i, j)))
		}
	}
}
func (matrix *DenseRealMatrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r = f(r, matrix.ConstAt(i, j))
		}
	}
	return r
}
func (matrix *DenseRealMatrix) ElementType() ScalarType {
	return RealType
}
func (matrix *DenseRealMatrix) Variables(order int) error {
	for i, _ := range matrix.values {
		if err := matrix.values[i].SetVariable(i, len(matrix.values), order); err != nil {
			return err
		}
	}
	return nil
}

/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) SwapRows(i, j int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	for k := 0; k < m; k++ {
		matrix.Swap(i, k, j, k)
	}
	return nil
}
func (matrix *DenseRealMatrix) SwapColumns(i, j int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	for k := 0; k < n; k++ {
		matrix.Swap(k, i, k, j)
	}
	return nil
}
func (matrix *DenseRealMatrix) PermuteRows(pi []int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	// permute matrix
	for i := 0; i < n; i++ {
		if pi[i] < 0 || pi[i] > n {
			return fmt.Errorf("SymmetricPermutation(): invalid permutation")
		}
		if i != pi[i] && pi[i] > i {
			matrix.SwapRows(i, pi[i])
		}
	}
	return nil
}
func (matrix *DenseRealMatrix) PermuteColumns(pi []int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	// permute matrix
	for i := 0; i < m; i++ {
		if pi[i] < 0 || pi[i] > n {
			return fmt.Errorf("SymmetricPermutation(): invalid permutation")
		}
		if i != pi[i] && pi[i] > i {
			matrix.SwapColumns(i, pi[i])
		}
	}
	return nil
}
func (matrix *DenseRealMatrix) SymmetricPermutation(pi []int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	for i := 0; i < n; i++ {
		if pi[i] < 0 || pi[i] > n {
			return fmt.Errorf("SymmetricPermutation(): invalid permutation")
		}
		if pi[i] > i {
			// permute rows
			matrix.SwapRows(i, pi[i])
			// permute colums
			matrix.SwapColumns(i, pi[i])
		}
	}
	return nil
}

/* type conversion
 * -------------------------------------------------------------------------- */
func (m *DenseRealMatrix) String() string {
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
func (a *DenseRealMatrix) Table() string {
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
func (m *DenseRealMatrix) Export(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	if _, err := fmt.Fprintf(w, "%s\n", m.Table()); err != nil {
		return err
	}
	return nil
}
func (m *DenseRealMatrix) Import(filename string) error {
	values := []float64{}
	rows := 0
	cols := 0
	var scanner *bufio.Scanner
	// open file
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	isgzip, err := isGzip(filename)
	if err != nil {
		return err
	}
	// check if file is gzipped
	if isgzip {
		g, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer g.Close()
		scanner = bufio.NewScanner(g)
	} else {
		scanner = bufio.NewScanner(f)
	}
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if cols == 0 {
			cols = len(fields)
		}
		if cols != len(fields) {
			return fmt.Errorf("invalid table")
		}
		for i := 0; i < len(fields); i++ {
			value, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				return fmt.Errorf("invalid table")
			}
			values = append(values, value)
		}
		rows++
	}
	*m = *NewDenseRealMatrix(rows, cols, values)
	return nil
}

/* json
 * -------------------------------------------------------------------------- */
func (obj *DenseRealMatrix) MarshalJSON() ([]byte, error) {
	if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
		n, m := obj.Dims()
		tmp := NullDenseRealMatrix(n, m)
		tmp.Set(obj)
		obj = tmp
	}
	r := struct {
		Values []*Real
		Rows   int
		Cols   int
	}{}
	r.Values = obj.values
	r.Rows = obj.rows
	r.Cols = obj.cols
	return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseRealMatrix) UnmarshalJSON(data []byte) error {
	r := struct {
		Values []*Real
		Rows   int
		Cols   int
	}{}
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	obj.values = nilDenseRealVector(len(r.Values))
	for i := 0; i < len(r.Values); i++ {
		obj.values[i] = r.Values[i]
	}
	obj.rows = r.Rows
	obj.rowMax = r.Rows
	obj.rowOffset = 0
	obj.cols = r.Cols
	obj.colMax = r.Cols
	obj.colOffset = 0
	obj.transposed = false
	obj.initTmp()
	return nil
}

/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *DenseRealMatrix) ConstIterator() MatrixConstIterator {
	return obj.ITERATOR()
}
func (obj *DenseRealMatrix) Iterator() MatrixIterator {
	return obj.ITERATOR()
}
func (obj *DenseRealMatrix) ITERATOR() *DenseRealMatrixIterator {
	r := DenseRealMatrixIterator{*obj.values.ITERATOR(), obj}
	return &r
}

/* iterator
 * -------------------------------------------------------------------------- */
type DenseRealMatrixIterator struct {
	DenseRealVectorIterator
	m *DenseRealMatrix
}

func (obj *DenseRealMatrixIterator) Index() (int, int) {
	return obj.m.ij(obj.DenseRealVectorIterator.Index())
}
func (obj *DenseRealMatrixIterator) Clone() *DenseRealMatrixIterator {
	return &DenseRealMatrixIterator{*obj.DenseRealVectorIterator.Clone(), obj.m}
}
func (obj *DenseRealMatrixIterator) CloneConstIterator() MatrixConstIterator {
	return &DenseRealMatrixIterator{*obj.DenseRealVectorIterator.Clone(), obj.m}
}
func (obj *DenseRealMatrixIterator) CloneIterator() MatrixIterator {
	return &DenseRealMatrixIterator{*obj.DenseRealVectorIterator.Clone(), obj.m}
}
