/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2020 Philipp Benner
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
/* -------------------------------------------------------------------------- */
package autodiff
/* -------------------------------------------------------------------------- */
import "fmt"
import "bytes"
import "bufio"
import "compress/gzip"
import "encoding/json"
import "io"
import "os"
import "strconv"
import "strings"
import "unsafe"
/* matrix type declaration
 * -------------------------------------------------------------------------- */
type SparseInt16Matrix struct {
  values *SparseInt16Vector
  rows int
  cols int
  rowOffset int
  rowMax int
  colOffset int
  colMax int
  tmp1 *SparseInt16Vector
  tmp2 *SparseInt16Vector
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewSparseInt16Matrix(rowIndices, colIndices []int, values []int16, rows, cols int) *SparseInt16Matrix {
  m := NullSparseInt16Matrix(rows, cols)
  if len(rowIndices) != len(colIndices) || len(colIndices) != len(values) {
    panic("number of row/col-indices does not match number of values")
  }
  for i := 0; i < len(colIndices); i++ {
    j1 := rowIndices[i]
    j2 := colIndices[i]
    if values[i] != 0.0 {
      m.At(j1, j2).SetInt16(values[i])
    }
  }
  return m
}
func NullSparseInt16Matrix(rows, cols int) *SparseInt16Matrix {
  m := SparseInt16Matrix{}
  m.values = NullSparseInt16Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  m.initTmp()
  return &m
}
func nilSparseInt16Matrix(rows, cols int) *SparseInt16Matrix {
  m := SparseInt16Matrix{}
  m.values = nilSparseInt16Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
}
func AsSparseInt16Matrix(matrix ConstMatrix) *SparseInt16Matrix {
  switch matrix_ := matrix.(type) {
  case *SparseInt16Matrix:
    return matrix_.Clone()
  }
  n, m := matrix.Dims()
  r := NullSparseInt16Matrix(n, m)
  for it := matrix.ConstIterator(); it.Ok(); it.Next() {
    i, j := it.Index()
    r.AT(i,j).Set(it.GetConst())
  }
  return r
}
func (matrix *SparseInt16Matrix) initTmp() {
  if matrix.tmp1 == nil || matrix.tmp1.Dim() < matrix.rows {
    matrix.tmp1 = NullSparseInt16Vector(matrix.rows)
  } else {
    matrix.tmp1 = matrix.tmp1.Slice(0, matrix.rows).(*SparseInt16Vector)
  }
  if matrix.tmp2 == nil || matrix.tmp2.Dim() < matrix.cols {
    matrix.tmp2 = NullSparseInt16Vector(matrix.cols)
  } else {
    matrix.tmp2 = matrix.tmp2.Slice(0, matrix.cols).(*SparseInt16Vector)
  }
}
/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *SparseInt16Matrix) Clone() *SparseInt16Matrix {
  return &SparseInt16Matrix{
    values : matrix.values.Clone(),
    rows : matrix.rows,
    cols : matrix.cols,
    rowOffset : matrix.rowOffset,
    rowMax : matrix.rowMax,
    colOffset : matrix.colOffset,
    colMax : matrix.colMax,
    tmp1 : matrix.tmp1.Clone(),
    tmp2 : matrix.tmp2.Clone() }
}
/* indexing
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) index(i, j int) int {
  if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
    panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
  }
  return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
}
func (matrix *SparseInt16Matrix) ij(k int) (int, int) {
  i := (k/matrix.colMax) - matrix.rowOffset
  j := (k%matrix.colMax) - matrix.colOffset
  return i, j
}
/* native matrix methods
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) AT(i, j int) Int16 {
  return matrix.values.AT(matrix.index(i, j))
}
func (matrix *SparseInt16Matrix) ROW(i int) *SparseInt16Vector {
  var v *SparseInt16Vector
  v = nilSparseInt16Vector(matrix.cols)
  for j := 0; j < matrix.cols; j++ {
    if s := matrix.values.AT_(matrix.index(i, j)); !s.nullScalar() {
      v.AT(j).SET(s)
    }
  }
  return v
}
func (matrix *SparseInt16Matrix) COL(j int) *SparseInt16Vector {
  var v *SparseInt16Vector
  v = nilSparseInt16Vector(matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    if s := matrix.values.AT_(matrix.index(i, j)); !s.nullScalar() {
      v.AT(i).SET(s)
    }
  }
  return v
}
func (matrix *SparseInt16Matrix) DIAG() *SparseInt16Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := nilSparseInt16Vector(n)
  for i := 0; i < n; i++ {
    if s := matrix.values.AT_(matrix.index(i, i)); !s.nullScalar() {
      v.AT(i).SET(s)
    }
  }
  return v
}
func (matrix *SparseInt16Matrix) SLICE(rfrom, rto, cfrom, cto int) *SparseInt16Matrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows = rto - rfrom
  m.colOffset += cfrom
  m.cols = cto - cfrom
  // crop tmp vectors
  m.initTmp()
  return &m
}
func (matrix *SparseInt16Matrix) AsSparseInt16Vector() *SparseInt16Vector {
  if matrix.cols < matrix.colMax - matrix.colOffset ||
    (matrix.rows < matrix.rowMax - matrix.rowOffset) {
    n, m := matrix.Dims()
    v := nilSparseInt16Vector(n*m)
    for it := matrix.ConstIterator(); it.Ok(); it.Next() {
      i, j := it.Index()
      v.At(i*matrix.cols + j).Set(matrix.ConstAt(i, j))
    }
    return v
  } else {
    return matrix.values
  }
}
/* matrix interface
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) CloneMatrix() Matrix {
  return matrix.Clone()
}
func (matrix *SparseInt16Matrix) At(i, j int) Scalar {
  return matrix.AT(i, j)
}
func (a *SparseInt16Matrix) Set(b ConstMatrix) {
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
func (matrix *SparseInt16Matrix) SetIdentity() {
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
func (matrix *SparseInt16Matrix) Reset() {
  for it := matrix.Iterator(); it.Ok(); it.Next() {
    it.Get().Reset()
  }
}
func (matrix *SparseInt16Matrix) Row(i int) Vector {
  return matrix.ROW(i)
}
func (matrix *SparseInt16Matrix) Col(j int) Vector {
  return matrix.COL(j)
}
func (matrix *SparseInt16Matrix) Diag() Vector {
  return matrix.DIAG()
}
func (matrix *SparseInt16Matrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *SparseInt16Matrix) Swap(i1, j1, i2, j2 int) {
  k1 := matrix.index(i1, j1)
  k2 := matrix.index(i2, j2)
  matrix.values.Swap(k1, k2)
}
func (matrix *SparseInt16Matrix) T() Matrix {
  m := &SparseInt16Matrix{
    values : NullSparseInt16Vector(matrix.values.Dim()),
    rows : matrix.cols,
    cols : matrix.rows,
    rowOffset : matrix.colOffset,
    rowMax : matrix.colMax,
    colOffset : matrix.rowOffset,
    colMax : matrix.rowMax,
    tmp1 : matrix.tmp2,
    tmp2 : matrix.tmp1 }
  for k1, value := range matrix.values.values {
    // transform indices so that iterators operate correctly
    i1, j1 := matrix.ij(k1)
    k2 := m.index(j1, i1)
    m.values.values[k2] = value
    m.values.indexInsert(k2)
  }
  return m
}
func (matrix *SparseInt16Matrix) Tip() {
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
        k = matrix.rows*k % (mn-1)
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
func (matrix *SparseInt16Matrix) AsVector() Vector {
  return matrix.AsSparseInt16Vector()
}
func (matrix *SparseInt16Matrix) storageLocation() uintptr {
  return uintptr(unsafe.Pointer(matrix.values.AT(0).ptr))
}
/* const interface
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) CloneConstMatrix() ConstMatrix {
  return matrix.Clone()
}
func (matrix *SparseInt16Matrix) Dims() (int, int) {
  if matrix == nil {
    return 0, 0
  } else {
    return matrix.rows, matrix.cols
  }
}
func (matrix *SparseInt16Matrix) Int8At(i, j int) int8 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetInt8()
}
func (matrix *SparseInt16Matrix) Int16At(i, j int) int16 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetInt16()
}
func (matrix *SparseInt16Matrix) Int32At(i, j int) int32 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetInt32()
}
func (matrix *SparseInt16Matrix) Int64At(i, j int) int64 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetInt64()
}
func (matrix *SparseInt16Matrix) IntAt(i, j int) int {
  return matrix.values.ConstAt(matrix.index(i, j)).GetInt()
}
func (matrix *SparseInt16Matrix) Float32At(i, j int) float32 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetFloat32()
}
func (matrix *SparseInt16Matrix) Float64At(i, j int) float64 {
  return matrix.values.ConstAt(matrix.index(i, j)).GetFloat64()
}
func (matrix *SparseInt16Matrix) ConstAt(i, j int) ConstScalar {
  return matrix.values.ConstAt(matrix.index(i, j))
}
func (matrix *SparseInt16Matrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *SparseInt16Matrix) ConstRow(i int) ConstVector {
  var v *SparseInt16Vector
  i = matrix.index(i, 0)
  v = matrix.values.Slice(i, i + matrix.cols).(*SparseInt16Vector)
  return v
}
func (matrix *SparseInt16Matrix) ConstCol(i int) ConstVector {
  return matrix.COL(i)
}
func (matrix *SparseInt16Matrix) ConstDiag() ConstVector {
  return matrix.DIAG()
}
func (matrix *SparseInt16Matrix) IsSymmetric(epsilon float64) bool {
  if n, m := matrix.Dims(); n != m {
    return false
  }
  for it := matrix.ConstIterator(); it.Ok(); it.Next() {
    i, j := it.Index()
    if !matrix.ConstAt(i,j).Equals(matrix.ConstAt(j,i), 1e-12) {
      return false
    }
  }
  return true
}
func (matrix *SparseInt16Matrix) AsConstVector() ConstVector {
  return matrix.AsSparseInt16Vector()
}
/* implement ScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) Map(f func(Scalar)) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      f(matrix.At(i, j))
    }
  }
}
func (matrix *SparseInt16Matrix) MapSet(f func(ConstScalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.ConstAt(i, j)))
    }
  }
}
func (matrix *SparseInt16Matrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.ConstAt(i, j))
    }
  }
  return r
}
func (matrix *SparseInt16Matrix) ElementType() ScalarType {
  return Int16Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *SparseInt16Matrix) SwapRows(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < m; k++ {
    matrix.Swap(i, k, j, k)
  }
  return nil
}
func (matrix *SparseInt16Matrix) SwapColumns(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < n; k++ {
    matrix.Swap(k, i, k, j)
  }
  return nil
}
func (matrix *SparseInt16Matrix) PermuteRows(pi []int) error {
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
func (matrix *SparseInt16Matrix) PermuteColumns(pi []int) error {
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
func (matrix *SparseInt16Matrix) SymmetricPermutation(pi []int) error {
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
func (m *SparseInt16Matrix) String() string {
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
      buffer.WriteString(m.ConstAt(i,j).String())
    }
    buffer.WriteString("]")
  }
  buffer.WriteString("]")
  return buffer.String()
}
func (a *SparseInt16Matrix) Table() string {
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
      buffer.WriteString(a.ConstAt(i,j).String())
    }
  }
  return buffer.String()
}
func (m *SparseInt16Matrix) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()
  w := bufio.NewWriter(f)
  defer w.Flush()
  if _, err := fmt.Fprintf(w, "%d %d\n", m.rows, m.cols); err != nil {
    return err
  }
  for it := m.ITERATOR(); it.Ok(); it.Next() {
    i, j := it.Index()
    if _, err := fmt.Fprintf(w, "%d %d %v\n", i, j, it.GET()); err != nil {
      return err
    }
  }
  return nil
}
func (m *SparseInt16Matrix) Import(filename string) error {
  rows := 0
  cols := 0
  rowIndices := []int{}
  colIndices := []int{}
  values := []int16{}
  var reader *bufio.Reader
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
    reader = bufio.NewReader(g)
  } else {
    reader = bufio.NewReader(f)
  }
  // scan header
  for i_ := 1;; i_++ {
    l, err := bufioReadLine(reader)
    if err == io.EOF {
      break
    }
    if err != nil {
      return err
    }
    if len(l) == 0 {
      continue
    }
    fields := strings.Fields(l)
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
    break
  }
  for i_ := 1;; i_++ {
    l, err := bufioReadLine(reader)
    if err == io.EOF {
      break
    }
    if err != nil {
      return err
    }
    if len(l) == 0 {
      continue
    }
    fields := strings.Fields(l)
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
      values = append(values, int16(v))
    }
  }
  *m = *NewSparseInt16Matrix(rowIndices, colIndices, values, rows, cols)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseInt16Matrix) MarshalJSON() ([]byte, error) {
  if obj.rowMax > obj.rows || obj.colMax > obj.cols {
    n, m := obj.Dims()
    tmp := NullSparseInt16Matrix(n, m)
    tmp.Set(obj)
    obj = tmp
  }
  k := []int{}
  v := []int16{}
  r := struct{Index []int; Value []int16; Rows int; Cols int}{}
  for it := obj.values.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, int16(it.GetConst().GetInt16()))
  }
  r.Index = k
  r.Value = v
  r.Rows = obj.rows
  r.Cols = obj.cols
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseInt16Matrix) UnmarshalJSON(data []byte) error {
  r := struct{Index []int; Value []int16; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  if len(r.Index) != len(r.Value) {
    return fmt.Errorf("invalid sparse vector")
  }
  obj.values = NewSparseInt16Vector(r.Index, r.Value, r.Rows*r.Cols)
  obj.rows = r.Rows
  obj.rowMax = r.Rows
  obj.rowOffset = 0
  obj.cols = r.Cols
  obj.colMax = r.Cols
  obj.colOffset = 0
  obj.initTmp()
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseInt16Matrix) ConstIterator() MatrixConstIterator {
  return obj.ITERATOR()
}
func (obj *SparseInt16Matrix) ConstIteratorFrom(i, j int) MatrixConstIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *SparseInt16Matrix) Iterator() MatrixIterator {
  return obj.ITERATOR()
}
func (obj *SparseInt16Matrix) IteratorFrom(i, j int) MatrixIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *SparseInt16Matrix) JointIterator(b ConstMatrix) MatrixJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseInt16Matrix) ITERATOR() *SparseInt16MatrixIterator {
  r := SparseInt16MatrixIterator{*obj.values.ITERATOR(), obj}
  return &r
}
func (obj *SparseInt16Matrix) ITERATOR_FROM(i, j int) *SparseInt16MatrixIterator {
  k := obj.index(i, j)
  r := SparseInt16MatrixIterator{*obj.values.ITERATOR_FROM(k), obj}
  return &r
}
func (obj *SparseInt16Matrix) JOINT_ITERATOR(b ConstMatrix) *SparseInt16MatrixJointIterator {
  r := SparseInt16MatrixJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, -1, Int16{}, nil}
  r.Next()
  return &r
}
func (obj *SparseInt16Matrix) JOINT3_ITERATOR(b, c ConstMatrix) *SparseInt16MatrixJoint3Iterator {
  r := SparseInt16MatrixJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, -1, Int16{}, nil, nil}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseInt16MatrixIterator struct {
  SparseInt16VectorIterator
  m *SparseInt16Matrix
}
func (obj *SparseInt16MatrixIterator) Index() (int, int) {
  return obj.m.ij(obj.SparseInt16VectorIterator.Index())
}
func (obj *SparseInt16MatrixIterator) Clone() *SparseInt16MatrixIterator {
  return &SparseInt16MatrixIterator{*obj.SparseInt16VectorIterator.Clone(), obj.m}
}
func (obj *SparseInt16MatrixIterator) CloneConstIterator() MatrixConstIterator {
  return &SparseInt16MatrixIterator{*obj.SparseInt16VectorIterator.Clone(), obj.m}
}
func (obj *SparseInt16MatrixIterator) CloneIterator() MatrixIterator {
  return &SparseInt16MatrixIterator{*obj.SparseInt16VectorIterator.Clone(), obj.m}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseInt16MatrixJointIterator struct {
  it1 *SparseInt16MatrixIterator
  it2 MatrixConstIterator
  i, j int
  s1 Int16
  s2 ConstScalar
}
func (obj *SparseInt16MatrixJointIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *SparseInt16MatrixJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt16() == int16(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt16() == int16(0))
}
func (obj *SparseInt16MatrixJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1.ptr = nil
  obj.s2 = nil
  if ok1 {
    obj.i, obj.j = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    i, j := obj.it2.Index()
    switch {
    case obj.i > i || (obj.i == i && obj.j > j) || !ok1:
      obj.i, obj.j = i, j
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GetConst()
    case obj.i == i && obj.j == j:
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1.ptr != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstInt16(0.0)
  }
}
func (obj *SparseInt16MatrixJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseInt16MatrixJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseInt16MatrixJointIterator) GET() (Int16, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseInt16MatrixJointIterator) Clone() *SparseInt16MatrixJointIterator {
  r := SparseInt16MatrixJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.i = obj.i
  r.j = obj.j
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *SparseInt16MatrixJointIterator) CloneJointIterator() MatrixJointIterator {
  return obj.Clone()
}
func (obj *SparseInt16MatrixJointIterator) CloneConstJointIterator() MatrixConstJointIterator {
  return obj.Clone()
}
/* joint3 iterator
 * -------------------------------------------------------------------------- */
type SparseInt16MatrixJoint3Iterator struct {
  it1 *SparseInt16MatrixIterator
  it2 MatrixConstIterator
  it3 MatrixConstIterator
  i, j int
  s1 Int16
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseInt16MatrixJoint3Iterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *SparseInt16MatrixJoint3Iterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt16() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetInt16() == 0.0) ||
         !(obj.s3 == nil || obj.s3.GetInt16() == 0.0)
}
func (obj *SparseInt16MatrixJoint3Iterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  ok3 := obj.it3.Ok()
  obj.s1.ptr = nil
  obj.s2 = nil
  obj.s3 = nil
  if ok1 {
    obj.i, obj.j = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    i, j := obj.it2.Index()
    switch {
    case obj.i > i || (obj.i == i && obj.j > j) || !ok1:
      obj.i = i
      obj.j = j
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GetConst()
    case obj.i == i && obj.j == j:
      obj.s2 = obj.it2.GetConst()
    }
  }
  if ok3 {
    i, j := obj.it3.Index()
    switch {
    case obj.i > i || (obj.i == i && obj.j > j) || (!ok1 && !ok2):
      obj.i = i
      obj.j = j
      obj.s1.ptr = nil
      obj.s2 = nil
      obj.s3 = obj.it3.GetConst()
    case obj.i == i && obj.j == j:
      obj.s3 = obj.it3.GetConst()
    }
  }
  if obj.s1.ptr != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstInt16(0.0)
  }
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstInt16(0.0)
  }
}
func (obj *SparseInt16MatrixJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2, obj.s3
  } else {
    return obj.s1, obj.s2, obj.s3
  }
}
func (obj *SparseInt16MatrixJoint3Iterator) GET() (Int16, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
