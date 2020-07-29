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
import "strconv"
import "strings"
import "os"
import "unsafe"
/* -------------------------------------------------------------------------- */
type DenseInt32Matrix struct {
  values []int32
  rows int
  cols int
  rowOffset int
  rowMax int
  colOffset int
  colMax int
  transposed bool
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseInt32Matrix(values []int32, rows, cols int) *DenseInt32Matrix {
  m := DenseInt32Matrix{}
  m.values = values
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
}
func NullDenseInt32Matrix(rows, cols int) *DenseInt32Matrix {
  m := DenseInt32Matrix{}
  m.values = make([]int32, rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
}
func AsDenseInt32Matrix(matrix ConstMatrix) *DenseInt32Matrix {
  switch matrix_ := matrix.(type) {
  case *DenseInt32Matrix:
    return matrix_.Clone()
  }
  n, m := matrix.Dims()
  r := NullDenseInt32Matrix(n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i,j).Set(matrix.ConstAt(i,j))
    }
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *DenseInt32Matrix) Clone() *DenseInt32Matrix {
  r := DenseInt32Matrix{}
  r = *matrix
  r.values = make([]int32, len(matrix.values))
  copy(r.values, matrix.values)
  return &r
}
/* indexing
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) index(i, j int) int {
  if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
    panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
  }
  if matrix.transposed {
    return (matrix.colOffset + j)*matrix.rowMax + (matrix.rowOffset + i)
  } else {
    return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
  }
}
func (matrix *DenseInt32Matrix) ij(k int) (int, int) {
  if matrix.transposed {
    i := (k%matrix.rowMax) - matrix.colOffset
    j := (k/matrix.rowMax) - matrix.rowOffset
    return i, j
  } else {
    i := (k/matrix.colMax) - matrix.rowOffset
    j := (k%matrix.colMax) - matrix.colOffset
    return i, j
  }
}
/* native matrix methods
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) AT(i, j int) Int32 {
  return Int32{&matrix.values[matrix.index(i, j)]}
}
func (matrix *DenseInt32Matrix) ROW(i int) DenseInt32Vector {
  v := make([]int32, matrix.cols)
  for j := 0; j < matrix.cols; j++ {
    v[j] = matrix.values[matrix.index(i, j)]
  }
  return DenseInt32Vector(v)
}
func (matrix *DenseInt32Matrix) COL(j int) DenseInt32Vector {
  v := make([]int32, matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    v[i] = matrix.values[matrix.index(i, j)]
  }
  return DenseInt32Vector(v)
}
func (matrix *DenseInt32Matrix) DIAG() DenseInt32Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := make([]int32, n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)]
  }
  return DenseInt32Vector(v)
}
func (matrix *DenseInt32Matrix) SLICE(rfrom, rto, cfrom, cto int) *DenseInt32Matrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows = rto - rfrom
  m.colOffset += cfrom
  m.cols = cto - cfrom
  return &m
}
/* matrix interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) CloneMatrix() Matrix {
  return matrix.Clone()
}
func (matrix *DenseInt32Matrix) At(i, j int) Scalar {
  return matrix.AT(i, j)
}
func (a *DenseInt32Matrix) Set(b ConstMatrix) {
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
func (matrix *DenseInt32Matrix) SetIdentity() {
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
func (matrix *DenseInt32Matrix) Reset() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i] = 0.0
  }
}
func (matrix *DenseInt32Matrix) Row(i int) Vector {
  return matrix.ROW(i)
}
func (matrix *DenseInt32Matrix) Col(j int) Vector {
  return matrix.COL(j)
}
func (matrix *DenseInt32Matrix) Diag() Vector {
  return matrix.DIAG()
}
func (matrix *DenseInt32Matrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *DenseInt32Matrix) Swap(i1, j1, i2, j2 int) {
  k1 := matrix.index(i1, j1)
  k2 := matrix.index(i2, j2)
  matrix.values[k1], matrix.values[k2] = matrix.values[k2], matrix.values[k1]
}
func (matrix *DenseInt32Matrix) T() Matrix {
  return &DenseInt32Matrix{
    values : matrix.values,
    rows : matrix.cols,
    cols : matrix.rows,
    transposed: !matrix.transposed,
    rowOffset : matrix.colOffset,
    rowMax : matrix.colMax,
    colOffset : matrix.rowOffset,
    colMax : matrix.rowMax }
}
func (matrix *DenseInt32Matrix) Tip() {
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
}
func (matrix *DenseInt32Matrix) AsVector() Vector {
  return DenseInt32Vector(matrix.values)
}
func (matrix *DenseInt32Matrix) storageLocation() uintptr {
  return uintptr(unsafe.Pointer(&matrix.values[0]))
}
/* const interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) CloneConstMatrix() ConstMatrix {
  return matrix.Clone()
}
func (matrix *DenseInt32Matrix) Dims() (int, int) {
  return matrix.rows, matrix.cols
}
func (matrix *DenseInt32Matrix) Int8At(i, j int) int8 {
  return int8(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) Int16At(i, j int) int16 {
  return int16(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) Int32At(i, j int) int32 {
  return int32(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) Int64At(i, j int) int64 {
  return int64(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) IntAt(i, j int) int {
  return int(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) Float32At(i, j int) float32 {
  return float32(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) Float64At(i, j int) float64 {
  return float64(matrix.values[matrix.index(i, j)])
}
func (matrix *DenseInt32Matrix) ConstAt(i, j int) ConstScalar {
  return Int32{&matrix.values[matrix.index(i, j)]}
}
func (matrix *DenseInt32Matrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows = rto - rfrom
  m.colOffset += cfrom
  m.cols = cto - cfrom
  return &m
}
func (matrix *DenseInt32Matrix) ConstRow(i int) ConstVector {
  var v []int32
  if matrix.transposed {
    v = make([]int32, matrix.cols)
    for j := 0; j < matrix.cols; j++ {
      v[j] = matrix.values[matrix.index(i, j)]
    }
  } else {
    i = matrix.index(i, 0)
    v = matrix.values[i:i + matrix.cols]
  }
  return DenseInt32Vector(v)
}
func (matrix *DenseInt32Matrix) ConstCol(j int) ConstVector {
  var v []int32
  if matrix.transposed {
    j = matrix.index(0, j)
    v = matrix.values[j:j + matrix.rows]
  } else {
    v = make([]int32, matrix.rows)
    for i := 0; i < matrix.rows; i++ {
      v[i] = matrix.values[matrix.index(i, j)]
    }
  }
  return DenseInt32Vector(v)
}
func (matrix *DenseInt32Matrix) ConstDiag() ConstVector {
  return matrix.DIAG()
}
func (matrix *DenseInt32Matrix) IsSymmetric(epsilon float64) bool {
  n, m := matrix.Dims()
  if n != m {
    return false
  }
  for i := 0; i < n; i++ {
    for j := i+1; j < m; j++ {
      if !matrix.ConstAt(i,j).Equals(matrix.ConstAt(j,i), 1e-12) {
        return false
      }
    }
  }
  return true
}
func (matrix *DenseInt32Matrix) AsConstVector() ConstVector {
  return DenseInt32Vector(matrix.values)
}
/* implement ScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) Map(f func(Scalar)) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      f(matrix.At(i, j))
    }
  }
}
func (matrix *DenseInt32Matrix) MapSet(f func(ConstScalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.ConstAt(i, j)))
    }
  }
}
func (matrix *DenseInt32Matrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.ConstAt(i, j))
    }
  }
  return r
}
func (matrix *DenseInt32Matrix) ElementType() ScalarType {
  return Int32Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *DenseInt32Matrix) SwapRows(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < m; k++ {
    matrix.Swap(i, k, j, k)
  }
  return nil
}
func (matrix *DenseInt32Matrix) SwapColumns(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < n; k++ {
    matrix.Swap(k, i, k, j)
  }
  return nil
}
func (matrix *DenseInt32Matrix) PermuteRows(pi []int) error {
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
func (matrix *DenseInt32Matrix) PermuteColumns(pi []int) error {
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
func (matrix *DenseInt32Matrix) SymmetricPermutation(pi []int) error {
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
func (m *DenseInt32Matrix) String() string {
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
func (a *DenseInt32Matrix) Table() string {
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
func (m *DenseInt32Matrix) Export(filename string) error {
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
func (m *DenseInt32Matrix) Import(filename string) error {
  values := []int32{}
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
      values = append(values, int32(value))
    }
    rows++
  }
  *m = *NewDenseInt32Matrix(values, rows, cols)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (a *DenseInt32Matrix) MarshalJSON() ([]byte, error) {
  if a.transposed || a.rowMax > a.rows || a.colMax > a.cols {
    n, m := a.Dims()
    tmp := NullDenseInt32Matrix(n, m)
    tmp.Set(a)
    a = tmp
  }
  r := struct{Values []int32; Rows int; Cols int}{}
  r.Values = a.values
  r.Rows = a.rows
  r.Cols = a.cols
  return json.MarshalIndent(r, "", "  ")
}
func (a *DenseInt32Matrix) UnmarshalJSON(data []byte) error {
  r := struct{Values []int32; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  a.values = r.Values
  a.rows = r.Rows
  a.rowMax = r.Rows
  a.rowOffset = 0
  a.cols = r.Cols
  a.colMax = r.Cols
  a.colOffset = 0
  a.transposed = false
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (m *DenseInt32Matrix) Iterator() MatrixIterator {
  return m.ITERATOR()
}
func (m *DenseInt32Matrix) IteratorFrom(i, j int) MatrixIterator {
  return m.ITERATOR_FROM(i, j)
}
func (m *DenseInt32Matrix) ConstIterator() MatrixConstIterator {
  return m.ITERATOR()
}
func (m *DenseInt32Matrix) ConstIteratorFrom(i, j int) MatrixConstIterator {
  return m.ITERATOR_FROM(i, j)
}
func (m *DenseInt32Matrix) JointIterator(b ConstMatrix) MatrixJointIterator {
  return m.JOINT_ITERATOR(b)
}
func (m *DenseInt32Matrix) ITERATOR() *DenseInt32MatrixIterator {
  r := DenseInt32MatrixIterator{m, 0, -1}
  r.Next()
  return &r
}
func (m *DenseInt32Matrix) ITERATOR_FROM(i, j int) *DenseInt32MatrixIterator {
  r := DenseInt32MatrixIterator{m, i, j-1}
  r.Next()
  return &r
}
func (m *DenseInt32Matrix) JOINT_ITERATOR(b ConstMatrix) *DenseInt32MatrixJointIterator {
  r := DenseInt32MatrixJointIterator{m.ITERATOR(), b.ConstIterator(), -1, -1, Int32{}, Int32{}}
  r.Next()
  return &r
}
/* const iterator
 * -------------------------------------------------------------------------- */
type DenseInt32MatrixIterator struct {
  m *DenseInt32Matrix
  i, j int
}
func (obj *DenseInt32MatrixIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseInt32MatrixIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseInt32MatrixIterator) GET() Int32 {
  return obj.m.AT(obj.i, obj.j)
}
func (obj *DenseInt32MatrixIterator) Ok() bool {
  return obj.i < obj.m.rowMax && obj.j < obj.m.colMax
}
func (obj *DenseInt32MatrixIterator) next() {
  if obj.j == obj.m.colMax-1 {
    obj.i = obj.i + 1
    obj.j = obj.m.colOffset
  } else {
    obj.j = obj.j + 1
  }
}
func (obj *DenseInt32MatrixIterator) Next() {
  obj.next()
  for obj.Ok() && obj.GET().nullScalar() {
    obj.next()
  }
}
func (obj *DenseInt32MatrixIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseInt32MatrixIterator) Clone() *DenseInt32MatrixIterator {
  return &DenseInt32MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseInt32MatrixIterator) CloneIterator() MatrixIterator {
  return &DenseInt32MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseInt32MatrixIterator) CloneConstIterator() MatrixConstIterator {
  return &DenseInt32MatrixIterator{obj.m, obj.i, obj.j}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseInt32MatrixJointIterator struct {
  it1 *DenseInt32MatrixIterator
  it2 MatrixConstIterator
  i, j int
  s1 Int32
  s2 ConstScalar
}
func (obj *DenseInt32MatrixJointIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseInt32MatrixJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt32() == int32(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt32() == int32(0))
}
func (obj *DenseInt32MatrixJointIterator) Next() {
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
    obj.s2 = ConstInt32(0.0)
  }
}
func (obj *DenseInt32MatrixJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt32MatrixJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt32MatrixJointIterator) GET() (Int32, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseInt32MatrixJointIterator) Clone() *DenseInt32MatrixJointIterator {
  r := DenseInt32MatrixJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.i = obj.i
  r.j = obj.j
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseInt32MatrixJointIterator) CloneJointIterator() MatrixJointIterator {
  return obj.Clone()
}
func (obj *DenseInt32MatrixJointIterator) CloneConstJointIterator() MatrixConstJointIterator {
  return obj.Clone()
}
