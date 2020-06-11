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
/* -------------------------------------------------------------------------- */
/* matrix type declaration
 * -------------------------------------------------------------------------- */
type SparseRealMatrix struct {
	values     *SparseRealVector
	rows       int
	cols       int
	rowOffset  int
	rowMax     int
	colOffset  int
	colMax     int
	transposed bool
	tmp1       *SparseRealVector
	tmp2       *SparseRealVector
}

/* constructors
 * -------------------------------------------------------------------------- */
func NewSparseRealMatrix(rows, cols int, rowIndices, colIndices []int, values []float64) *SparseRealMatrix {
	m := NullSparseRealMatrix(rows, cols)
	if len(rowIndices) != len(colIndices) || len(colIndices) != len(values) {
		panic("number of row/col-indices does not match number of values")
	}
	for i := 0; i < len(colIndices); i++ {
		j1 := rowIndices[i]
		j2 := colIndices[i]
		if values[i] != 0.0 {
			m.At(j1, j2).SetValue(values[i])
		}
	}
	return m
}
func NullSparseRealMatrix(rows, cols int) *SparseRealMatrix {
	m := SparseRealMatrix{}
	m.values = NullSparseRealVector(rows * cols)
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	m.initTmp()
	return &m
}
func nilSparseRealMatrix(rows, cols int) *SparseRealMatrix {
	m := SparseRealMatrix{}
	m.values = nilSparseRealVector(rows * cols)
	m.rows = rows
	m.cols = cols
	m.rowOffset = 0
	m.rowMax = rows
	m.colOffset = 0
	m.colMax = cols
	return &m
}
func AsSparseRealMatrix(matrix ConstMatrix) *SparseRealMatrix {
	switch matrix_ := matrix.(type) {
	case *SparseRealMatrix:
		return matrix_.Clone()
	}
	n, m := matrix.Dims()
	r := NullSparseRealMatrix(n, m)
	for it := matrix.ConstIterator(); it.Ok(); it.Next() {
		i, j := it.Index()
		r.AT(i, j).Set(it.GetConst())
	}
	return r
}
func (matrix *SparseRealMatrix) initTmp() {
	if matrix.tmp1 == nil || matrix.tmp1.Dim() < matrix.rows {
		matrix.tmp1 = NullSparseRealVector(matrix.rows)
	} else {
		matrix.tmp1 = matrix.tmp1.Slice(0, matrix.rows).(*SparseRealVector)
	}
	if matrix.tmp2 == nil || matrix.tmp2.Dim() < matrix.cols {
		matrix.tmp2 = NullSparseRealVector(matrix.cols)
	} else {
		matrix.tmp2 = matrix.tmp2.Slice(0, matrix.cols).(*SparseRealVector)
	}
}

/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *SparseRealMatrix) Clone() *SparseRealMatrix {
	return &SparseRealMatrix{
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
func (matrix *SparseRealMatrix) CloneMatrix() Matrix {
	return matrix.Clone()
}
func (matrix *SparseRealMatrix) CloneConstMatrix() ConstMatrix {
	return matrix.Clone()
}

/* field access
 * -------------------------------------------------------------------------- */
func (matrix *SparseRealMatrix) index(i, j int) int {
	if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
		panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
	}
	if matrix.transposed {
		return (matrix.colOffset+j)*matrix.rowMax + (matrix.rowOffset + i)
	} else {
		return (matrix.rowOffset+i)*matrix.colMax + (matrix.colOffset + j)
	}
}
func (matrix *SparseRealMatrix) ij(k int) (int, int) {
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
func (matrix *SparseRealMatrix) Dims() (int, int) {
	if matrix == nil {
		return 0, 0
	} else {
		return matrix.rows, matrix.cols
	}
}
func (matrix *SparseRealMatrix) Row(i int) Vector {
	return matrix.ROW(i)
}
func (matrix *SparseRealMatrix) ROW(i int) *SparseRealVector {
	var v *SparseRealVector
	if matrix.transposed {
		v = nilSparseRealVector(matrix.cols)
		for j := 0; j < matrix.cols; j++ {
			v.At(j).Set(matrix.values.ConstAt(matrix.index(i, j)))
		}
	} else {
		i = matrix.index(i, 0)
		v = matrix.values.Slice(i, i+matrix.cols).(*SparseRealVector)
	}
	return v
}
func (matrix *SparseRealMatrix) Col(j int) Vector {
	return matrix.COL(j)
}
func (matrix *SparseRealMatrix) COL(j int) *SparseRealVector {
	var v *SparseRealVector
	if matrix.transposed {
		j = matrix.index(0, j)
		v = matrix.values.Slice(j, j+matrix.rows).(*SparseRealVector)
	} else {
		v = nilSparseRealVector(matrix.rows)
		for i := 0; i < matrix.rows; i++ {
			v.At(i).Set(matrix.values.ConstAt(matrix.index(i, j)))
		}
	}
	return v
}
func (matrix *SparseRealMatrix) Diag() Vector {
	return matrix.DIAG()
}
func (matrix *SparseRealMatrix) DIAG() *SparseRealVector {
	n, m := matrix.Dims()
	if n != m {
		panic("Diag(): not a square matrix!")
	}
	v := nilSparseRealVector(n)
	for i := 0; i < n; i++ {
		v.At(i).Set(matrix.values.ConstAt(matrix.index(i, i)))
	}
	return v
}
func (matrix *SparseRealMatrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
	m := *matrix
	m.rowOffset += rfrom
	m.rows = rto - rfrom
	m.colOffset += cfrom
	m.cols = cto - cfrom
	// crop tmp vectors
	m.initTmp()
	return &m
}
func (matrix *SparseRealMatrix) Swap(i1, j1, i2, j2 int) {
	k1 := matrix.index(i1, j1)
	k2 := matrix.index(i2, j2)
	matrix.values.Swap(k1, k2)
}
func (matrix *SparseRealMatrix) AsVector() Vector {
	return matrix.AsSparseRealVector()
}
func (matrix *SparseRealMatrix) AsConstVector() ConstVector {
	return matrix.AsVector()
}
func (matrix *SparseRealMatrix) AsSparseRealVector() *SparseRealVector {
	if matrix.cols < matrix.colMax-matrix.colOffset ||
		(matrix.rows < matrix.rowMax-matrix.rowOffset) {
		n, m := matrix.Dims()
		v := nilSparseRealVector(n * m)
		for it := matrix.ConstIterator(); it.Ok(); it.Next() {
			i, j := it.Index()
			v.At(i*matrix.cols + j).Set(matrix.ConstAt(i, j))
		}
		return v
	} else {
		return matrix.values
	}
}

/* -------------------------------------------------------------------------- */
func (matrix *SparseRealMatrix) T() Matrix {
	return &SparseRealMatrix{
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
func (matrix *SparseRealMatrix) Tip() {
	mn := matrix.values.Dim()
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
			matrix.values.Swap(k, cycle)
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
func (matrix *SparseRealMatrix) ValueAt(i, j int) float64 {
	return matrix.values.ConstAt(matrix.index(i, j)).GetValue()
}
func (matrix *SparseRealMatrix) ConstAt(i, j int) ConstScalar {
	return matrix.values.ConstAt(matrix.index(i, j))
}
func (matrix *SparseRealMatrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
	return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *SparseRealMatrix) ConstRow(i int) ConstVector {
	return matrix.ROW(i)
}
func (matrix *SparseRealMatrix) ConstCol(i int) ConstVector {
	return matrix.COL(i)
}
func (matrix *SparseRealMatrix) ConstDiag() ConstVector {
	return matrix.DIAG()
}
func (matrix *SparseRealMatrix) GetValues() []float64 {
	n, m := matrix.Dims()
	s := make([]float64, n*m)
	for it := matrix.ConstIterator(); it.Ok(); it.Next() {
		i, j := it.Index()
		s[i*m+j] = matrix.ConstAt(i, j).GetValue()
	}
	return s
}

/* -------------------------------------------------------------------------- */
func (matrix *SparseRealMatrix) At(i, j int) Scalar {
	return matrix.AT(i, j)
}
func (matrix *SparseRealMatrix) AT(i, j int) *Real {
	return matrix.values.AT(matrix.index(i, j))
}
func (matrix *SparseRealMatrix) Reset() {
	for it := matrix.Iterator(); it.Ok(); it.Next() {
		it.Get().Reset()
	}
}
func (matrix *SparseRealMatrix) ResetDerivatives() {
	for it := matrix.Iterator(); it.Ok(); it.Next() {
		it.Get().ResetDerivatives()
	}
}
func (a *SparseRealMatrix) Set(b ConstMatrix) {
	n1, m1 := a.Dims()
	n2, m2 := b.Dims()
	if n1 != n2 || m1 != m2 {
		panic("Copy(): Matrix dimension does not match!")
	}
	for it := a.Iterator(); it.Ok(); it.Next() {
		i, j := it.Index()
		it.Get().Set(b.ConstAt(i, j))
	}
}
func (matrix *SparseRealMatrix) SetIdentity() {
	c := NewScalar(matrix.ElementType(), 1.0)
	for it := matrix.Iterator(); it.Ok(); it.Next() {
		i, j := it.Index()
		if i == j {
			it.Get().Set(c)
		} else {
			it.Get().Reset()
		}
	}
}
func (matrix *SparseRealMatrix) IsSymmetric(epsilon float64) bool {
	if n, m := matrix.Dims(); n != m {
		return false
	}
	for it := matrix.ConstIterator(); it.Ok(); it.Next() {
		i, j := it.Index()
		if !matrix.ConstAt(i, j).Equals(matrix.ConstAt(j, i), 1e-12) {
			return false
		}
	}
	return true
}
func (matrix *SparseRealMatrix) storageLocation() uintptr {
	return uintptr(unsafe.Pointer(matrix.values.AT(0)))
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *SparseRealMatrix) Map(f func(Scalar)) {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			f(matrix.At(i, j))
		}
	}
}
func (matrix *SparseRealMatrix) MapSet(f func(ConstScalar) Scalar) {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			matrix.At(i, j).Set(f(matrix.ConstAt(i, j)))
		}
	}
}
func (matrix *SparseRealMatrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	n, m := matrix.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			r = f(r, matrix.ConstAt(i, j))
		}
	}
	return r
}
func (matrix *SparseRealMatrix) ElementType() ScalarType {
	return RealType
}
func (matrix *SparseRealMatrix) Variables(order int) error {
	for i, v := range matrix.values.values {
		if err := v.SetVariable(i, matrix.values.Dim(), order); err != nil {
			return err
		}
	}
	return nil
}

/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *SparseRealMatrix) SwapRows(i, j int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	for k := 0; k < m; k++ {
		matrix.Swap(i, k, j, k)
	}
	return nil
}
func (matrix *SparseRealMatrix) SwapColumns(i, j int) error {
	n, m := matrix.Dims()
	if n != m {
		return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
	}
	for k := 0; k < n; k++ {
		matrix.Swap(k, i, k, j)
	}
	return nil
}
func (matrix *SparseRealMatrix) PermuteRows(pi []int) error {
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
func (matrix *SparseRealMatrix) PermuteColumns(pi []int) error {
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
func (matrix *SparseRealMatrix) SymmetricPermutation(pi []int) error {
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
func (m *SparseRealMatrix) String() string {
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
func (a *SparseRealMatrix) Table() string {
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
func (m *SparseRealMatrix) Export(filename string) error {
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
func (m *SparseRealMatrix) Import(filename string) error {
	rows := 0
	cols := 0
	rowIndices := []int{}
	colIndices := []int{}
	values := []float64{}
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
	// scan header
	if scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			return fmt.Errorf("invalid sparse matrix format")
		}
		if v, err := strconv.ParseInt(fields[0], 10, 64); err != nil {
			return err
		} else {
			rows = int(v)
		}
		if v, err := strconv.ParseInt(fields[1], 10, 64); err != nil {
			return err
		} else {
			cols = int(v)
		}
	}
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 3 {
			return fmt.Errorf("invalid sparse matrix format")
		}
		if v, err := strconv.ParseInt(fields[0], 10, 64); err != nil {
			return err
		} else {
			rowIndices = append(rowIndices, int(v))
		}
		if v, err := strconv.ParseInt(fields[1], 10, 64); err != nil {
			return err
		} else {
			colIndices = append(colIndices, int(v))
		}
		if v, err := strconv.ParseFloat(fields[2], 64); err != nil {
			return err
		} else {
			values = append(values, v)
		}
	}
	*m = *NewSparseRealMatrix(rows, cols, rowIndices, colIndices, values)
	return nil
}

/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseRealMatrix) MarshalJSON() ([]byte, error) {
	if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
		n, m := obj.Dims()
		tmp := NullSparseRealMatrix(n, m)
		tmp.Set(obj)
		obj = tmp
	}
	k := []int{}
	v := []float64{}
	r := struct {
		Index []int
		Value []float64
		Rows  int
		Cols  int
	}{}
	for it := obj.values.ConstIterator(); it.Ok(); it.Next() {
		k = append(k, it.Index())
		v = append(v, it.GetValue())
	}
	r.Index = k
	r.Value = v
	r.Rows = obj.rows
	r.Cols = obj.cols
	return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseRealMatrix) UnmarshalJSON(data []byte) error {
	r := struct {
		Index []int
		Value []float64
		Rows  int
		Cols  int
	}{}
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	if len(r.Index) != len(r.Value) {
		return fmt.Errorf("invalid sparse vector")
	}
	obj.values = NewSparseRealVector(r.Index, r.Value, r.Rows*r.Cols)
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
func (obj *SparseRealMatrix) ConstIterator() MatrixConstIterator {
	return obj.ITERATOR()
}
func (obj *SparseRealMatrix) Iterator() MatrixIterator {
	return obj.ITERATOR()
}
func (obj *SparseRealMatrix) ITERATOR() *SparseRealMatrixIterator {
	r := SparseRealMatrixIterator{*obj.values.ITERATOR(), obj}
	return &r
}

/* iterator
 * -------------------------------------------------------------------------- */
type SparseRealMatrixIterator struct {
	SparseRealVectorIterator
	m *SparseRealMatrix
}

func (obj *SparseRealMatrixIterator) Index() (int, int) {
	return obj.m.ij(obj.SparseRealVectorIterator.Index())
}
func (obj *SparseRealMatrixIterator) Clone() *SparseRealMatrixIterator {
	return &SparseRealMatrixIterator{*obj.SparseRealVectorIterator.Clone(), obj.m}
}
func (obj *SparseRealMatrixIterator) CloneConstIterator() MatrixConstIterator {
	return &SparseRealMatrixIterator{*obj.SparseRealVectorIterator.Clone(), obj.m}
}
func (obj *SparseRealMatrixIterator) CloneIterator() MatrixIterator {
	return &SparseRealMatrixIterator{*obj.SparseRealVectorIterator.Clone(), obj.m}
}
