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
type DenseReal32Matrix struct {
  values DenseReal32Vector
  rows int
  cols int
  rowOffset int
  rowMax int
  colOffset int
  colMax int
  transposed bool
  tmp1 DenseReal32Vector
  tmp2 DenseReal32Vector
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseReal32Matrix(values []float32, rows, cols int) *DenseReal32Matrix {
  m := nilDenseReal32Matrix(rows, cols)
  v := m.values
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      v[i] = NewReal32(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      v[i] = NewReal32(values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  m.initTmp()
  return m
}
func NullDenseReal32Matrix(rows, cols int) *DenseReal32Matrix {
  m := DenseReal32Matrix{}
  m.values = NullDenseReal32Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  m.initTmp()
  return &m
}
func nilDenseReal32Matrix(rows, cols int) *DenseReal32Matrix {
  m := DenseReal32Matrix{}
  m.values = nilDenseReal32Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
}
func AsDenseReal32Matrix(matrix ConstMatrix) *DenseReal32Matrix {
  switch matrix_ := matrix.(type) {
  case *DenseReal32Matrix:
    return matrix_.Clone()
  }
  n, m := matrix.Dims()
  r := NullDenseReal32Matrix(n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i,j).Set(matrix.ConstAt(i,j))
    }
  }
  return r
}
func (matrix *DenseReal32Matrix) initTmp() {
  if len(matrix.tmp1) < matrix.rows {
    matrix.tmp1 = NullDenseReal32Vector(matrix.rows)
  } else {
    matrix.tmp1 = matrix.tmp1[0:matrix.rows]
  }
  if len(matrix.tmp2) < matrix.cols {
    matrix.tmp2 = NullDenseReal32Vector(matrix.cols)
  } else {
    matrix.tmp2 = matrix.tmp2[0:matrix.cols]
  }
}
/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *DenseReal32Matrix) Clone() *DenseReal32Matrix {
  return &DenseReal32Matrix{
    values : matrix.values.Clone(),
    rows : matrix.rows,
    cols : matrix.cols,
    transposed: matrix.transposed,
    rowOffset : matrix.rowOffset,
    rowMax : matrix.rowMax,
    colOffset : matrix.colOffset,
    colMax : matrix.colMax,
    tmp1 : matrix.tmp1.Clone(),
    tmp2 : matrix.tmp2.Clone() }
}
/* indexing
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) index(i, j int) int {
  if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
    panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
  }
  if matrix.transposed {
    return (matrix.colOffset + j)*matrix.rowMax + (matrix.rowOffset + i)
  } else {
    return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
  }
}
func (matrix *DenseReal32Matrix) ij(k int) (int, int) {
  if matrix.transposed {
    i := (k%matrix.colMax) - matrix.colOffset
    j := (k/matrix.colMax) - matrix.rowOffset
    return i, j
  } else {
    i := (k/matrix.rowMax) - matrix.rowOffset
    j := (k%matrix.rowMax) - matrix.colOffset
    return i, j
  }
}
/* native matrix methods
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) AT(i, j int) *Real32 {
  return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseReal32Matrix) ROW(i int) DenseReal32Vector {
  v := nilDenseReal32Vector(matrix.cols)
  for j := 0; j < matrix.cols; j++ {
    v[j] = matrix.values[matrix.index(i, j)].Clone()
  }
  return v
}
func (matrix *DenseReal32Matrix) COL(j int) DenseReal32Vector {
  v := nilDenseReal32Vector(matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    v[i] = matrix.values[matrix.index(i, j)].Clone()
  }
  return v
}
func (matrix *DenseReal32Matrix) DIAG() DenseReal32Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := nilDenseReal32Vector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)].Clone()
  }
  return v
}
func (matrix *DenseReal32Matrix) SLICE(rfrom, rto, cfrom, cto int) *DenseReal32Matrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows = rto - rfrom
  m.colOffset += cfrom
  m.cols = cto - cfrom
  // crop tmp vectors
  m.initTmp()
  return &m
}
func (matrix *DenseReal32Matrix) AsDenseReal32Vector() DenseReal32Vector {
  if matrix.cols < matrix.colMax - matrix.colOffset ||
    (matrix.rows < matrix.rowMax - matrix.rowOffset) {
    n, m := matrix.Dims()
    v := nilDenseReal32Vector(n*m)
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        v[i*matrix.cols + j] = matrix.AT(i, j)
      }
    }
    return v
  } else {
    return DenseReal32Vector(matrix.values)
  }
}
/* matrix interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) CloneMatrix() Matrix {
  return matrix.Clone()
}
func (matrix *DenseReal32Matrix) At(i, j int) Scalar {
  return matrix.AT(i, j)
}
func (a *DenseReal32Matrix) Set(b ConstMatrix) {
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
func (matrix *DenseReal32Matrix) SetIdentity() {
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
func (matrix *DenseReal32Matrix) Reset() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].Reset()
  }
}
func (matrix *DenseReal32Matrix) Row(i int) Vector {
  return matrix.ROW(i)
}
func (matrix *DenseReal32Matrix) Col(j int) Vector {
  return matrix.COL(j)
}
func (matrix *DenseReal32Matrix) Diag() Vector {
  return matrix.DIAG()
}
func (matrix *DenseReal32Matrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal32Matrix) Swap(i1, j1, i2, j2 int) {
  k1 := matrix.index(i1, j1)
  k2 := matrix.index(i2, j2)
  matrix.values[k1], matrix.values[k2] = matrix.values[k2], matrix.values[k1]
}
func (matrix *DenseReal32Matrix) T() Matrix {
  return matrix.MagicT()
}
func (matrix *DenseReal32Matrix) Tip() {
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
        k = matrix.rows*k % (mn-1)
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
func (matrix *DenseReal32Matrix) AsVector() Vector {
  return matrix.AsDenseReal32Vector()
}
func (matrix *DenseReal32Matrix) storageLocation() uintptr {
  return uintptr(unsafe.Pointer(&matrix.values[0]))
}
/* const interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) CloneConstMatrix() ConstMatrix {
  return matrix.Clone()
}
func (matrix *DenseReal32Matrix) Dims() (int, int) {
  if matrix == nil {
    return 0, 0
  } else {
    return matrix.rows, matrix.cols
  }
}
func (matrix *DenseReal32Matrix) Int8At(i, j int) int8 {
  return matrix.values[matrix.index(i, j)].GetInt8()
}
func (matrix *DenseReal32Matrix) Int16At(i, j int) int16 {
  return matrix.values[matrix.index(i, j)].GetInt16()
}
func (matrix *DenseReal32Matrix) Int32At(i, j int) int32 {
  return matrix.values[matrix.index(i, j)].GetInt32()
}
func (matrix *DenseReal32Matrix) Int64At(i, j int) int64 {
  return matrix.values[matrix.index(i, j)].GetInt64()
}
func (matrix *DenseReal32Matrix) IntAt(i, j int) int {
  return matrix.values[matrix.index(i, j)].GetInt()
}
func (matrix *DenseReal32Matrix) Float32At(i, j int) float32 {
  return matrix.values[matrix.index(i, j)].GetFloat32()
}
func (matrix *DenseReal32Matrix) Float64At(i, j int) float64 {
  return matrix.values[matrix.index(i, j)].GetFloat64()
}
func (matrix *DenseReal32Matrix) ConstAt(i, j int) ConstScalar {
  return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseReal32Matrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal32Matrix) ConstRow(i int) ConstVector {
  // no cloning required...
  var v DenseReal32Vector
  if matrix.transposed {
    v = nilDenseReal32Vector(matrix.cols)
    for j := 0; j < matrix.cols; j++ {
      v[j] = matrix.values[matrix.index(i, j)]
    }
  } else {
    i = matrix.index(i, 0)
    v = matrix.values[i:i + matrix.cols]
  }
  return v
}
func (matrix *DenseReal32Matrix) ConstCol(j int) ConstVector {
  // no cloning required...
  var v DenseReal32Vector
  if matrix.transposed {
    j = matrix.index(0, j)
    v = matrix.values[j:j + matrix.rows]
  } else {
    v = nilDenseReal32Vector(matrix.rows)
    for i := 0; i < matrix.rows; i++ {
      v[i] = matrix.values[matrix.index(i, j)]
    }
  }
  return v
}
func (matrix *DenseReal32Matrix) ConstDiag() ConstVector {
  // no cloning required...
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := nilDenseReal32Vector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)]
  }
  return v
}
func (matrix *DenseReal32Matrix) IsSymmetric(epsilon float64) bool {
  n, m := matrix.Dims()
  if n != m {
    return false
  }
  for i := 0; i < n; i++ {
    for j := i+1; j < m; j++ {
      if !matrix.At(i,j).Equals(matrix.At(j,i), 1e-12) {
        return false
      }
    }
  }
  return true
}
func (matrix *DenseReal32Matrix) AsConstVector() ConstVector {
  return matrix.AsDenseReal32Vector()
}
/* magic interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) CloneMagicMatrix() MagicMatrix {
  return matrix.Clone()
}
func (matrix *DenseReal32Matrix) MagicAt(i, j int) MagicScalar {
  return matrix.AT(i, j)
}
func (matrix *DenseReal32Matrix) MagicSlice(rfrom, rto, cfrom, cto int) MagicMatrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal32Matrix) MagicT() MagicMatrix {
  return &DenseReal32Matrix{
    values : matrix.values,
    rows : matrix.cols,
    cols : matrix.rows,
    transposed: !matrix.transposed,
    rowOffset : matrix.colOffset,
    rowMax : matrix.colMax,
    colOffset : matrix.rowOffset,
    colMax : matrix.rowMax,
    tmp1 : matrix.tmp2,
    tmp2 : matrix.tmp1 }
}
func (matrix *DenseReal32Matrix) ResetDerivatives() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].ResetDerivatives()
  }
}
func (matrix *DenseReal32Matrix) AsMagicVector() MagicVector {
  return matrix.AsDenseReal32Vector()
}
/* implement MagicScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) Map(f func(Scalar)) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      f(matrix.At(i, j))
    }
  }
}
func (matrix *DenseReal32Matrix) MapSet(f func(ConstScalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.ConstAt(i, j)))
    }
  }
}
func (matrix *DenseReal32Matrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.ConstAt(i, j))
    }
  }
  return r
}
func (matrix *DenseReal32Matrix) ElementType() ScalarType {
  return Real32Type
}
// Treat all elements as variables for automatic differentiation. This method should only be called on a single vector or matrix. If multiple matrices should be treated as variables, then a single matrix must be allocated first and sliced after calling this method.
func (matrix *DenseReal32Matrix) Variables(order int) error {
  for i, _ := range matrix.values {
    if err := matrix.values[i].SetVariable(i, len(matrix.values), order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal32Matrix) SwapRows(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < m; k++ {
    matrix.Swap(i, k, j, k)
  }
  return nil
}
func (matrix *DenseReal32Matrix) SwapColumns(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < n; k++ {
    matrix.Swap(k, i, k, j)
  }
  return nil
}
func (matrix *DenseReal32Matrix) PermuteRows(pi []int) error {
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
func (matrix *DenseReal32Matrix) PermuteColumns(pi []int) error {
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
func (matrix *DenseReal32Matrix) SymmetricPermutation(pi []int) error {
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
func (m *DenseReal32Matrix) String() string {
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
func (a *DenseReal32Matrix) Table() string {
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
func (m *DenseReal32Matrix) Export(filename string) error {
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
func (m *DenseReal32Matrix) Import(filename string) error {
  values := []float32{}
  rows := 0
  cols := 0
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
      values = append(values, float32(value))
    }
    rows++
  }
  *m = *NewDenseReal32Matrix(values, rows, cols)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *DenseReal32Matrix) MarshalJSON() ([]byte, error) {
  if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
    n, m := obj.Dims()
    tmp := NullDenseReal32Matrix(n, m)
    tmp.Set(obj)
    obj = tmp
  }
  r := struct{Values []*Real32; Rows int; Cols int}{}
  r.Values = obj.values
  r.Rows = obj.rows
  r.Cols = obj.cols
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseReal32Matrix) UnmarshalJSON(data []byte) error {
  r := struct{Values []*Real32; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  obj.values = nilDenseReal32Vector(len(r.Values))
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
func (obj *DenseReal32Matrix) ConstIterator() MatrixConstIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal32Matrix) ConstIteratorFrom(i, j int) MatrixConstIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal32Matrix) MagicIterator() MatrixMagicIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal32Matrix) MagicIteratorFrom(i, j int) MatrixMagicIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal32Matrix) Iterator() MatrixIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal32Matrix) IteratorFrom(i, j int) MatrixIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal32Matrix) JointIterator(b ConstMatrix) MatrixJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *DenseReal32Matrix) ITERATOR() *DenseReal32MatrixIterator {
  r := DenseReal32MatrixIterator{obj, 0, -1}
  r.Next()
  return &r
}
func (obj *DenseReal32Matrix) ITERATOR_FROM(i, j int) *DenseReal32MatrixIterator {
  r := DenseReal32MatrixIterator{obj, i, j-1}
  r.Next()
  return &r
}
func (obj *DenseReal32Matrix) JOINT_ITERATOR(b ConstMatrix) *DenseReal32MatrixJointIterator {
  r := DenseReal32MatrixJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, -1, nil, nil}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type DenseReal32MatrixIterator struct {
  m *DenseReal32Matrix
  i, j int
}
func (obj *DenseReal32MatrixIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseReal32MatrixIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseReal32MatrixIterator) GetMagic() MagicScalar {
  return obj.GET()
}
func (obj *DenseReal32MatrixIterator) GET() *Real32 {
  return obj.m.AT(obj.i, obj.j)
}
func (obj *DenseReal32MatrixIterator) Ok() bool {
  return obj.i < obj.m.rowMax && obj.j < obj.m.colMax
}
func (obj *DenseReal32MatrixIterator) next() {
  if obj.j == obj.m.cols-1 {
    obj.i = obj.i + 1
    obj.j = 0
  } else {
    obj.j = obj.j + 1
  }
}
func (obj *DenseReal32MatrixIterator) Next() {
  obj.next()
  for obj.Ok() && obj.GET().nullScalar() {
    obj.next()
  }
}
func (obj *DenseReal32MatrixIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseReal32MatrixIterator) Clone() *DenseReal32MatrixIterator {
  return &DenseReal32MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal32MatrixIterator) CloneIterator() MatrixIterator {
  return &DenseReal32MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal32MatrixIterator) CloneConstIterator() MatrixConstIterator {
  return &DenseReal32MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal32MatrixIterator) CloneMagicIterator() MatrixMagicIterator {
  return &DenseReal32MatrixIterator{obj.m, obj.i, obj.j}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal32MatrixJointIterator struct {
  it1 *DenseReal32MatrixIterator
  it2 MatrixConstIterator
  i, j int
  s1 *Real32
  s2 ConstScalar
}
func (obj *DenseReal32MatrixJointIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseReal32MatrixJointIterator) Ok() bool {
  return !(obj.s1 == nil || obj.s1.GetFloat32() == float32(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat32() == float32(0))
}
func (obj *DenseReal32MatrixJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = nil
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
      obj.s1 = nil
      obj.s2 = obj.it2.GetConst()
    case obj.i == i && obj.j == j:
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1 != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstFloat32(0.0)
  }
}
func (obj *DenseReal32MatrixJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal32MatrixJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal32MatrixJointIterator) GET() (*Real32, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseReal32MatrixJointIterator) Clone() *DenseReal32MatrixJointIterator {
  r := DenseReal32MatrixJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.i = obj.i
  r.j = obj.j
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseReal32MatrixJointIterator) CloneJointIterator() MatrixJointIterator {
  return obj.Clone()
}
func (obj *DenseReal32MatrixJointIterator) CloneConstJointIterator() MatrixConstJointIterator {
  return obj.Clone()
}
