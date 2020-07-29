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
/* matrix type declaration
 * -------------------------------------------------------------------------- */
type DenseReal64Matrix struct {
  values DenseReal64Vector
  rows int
  cols int
  rowOffset int
  rowMax int
  colOffset int
  colMax int
  transposed bool
  tmp1 DenseReal64Vector
  tmp2 DenseReal64Vector
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseReal64Matrix(values []float64, rows, cols int) *DenseReal64Matrix {
  m := nilDenseReal64Matrix(rows, cols)
  v := m.values
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      v[i] = NewReal64(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      v[i] = NewReal64(values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  m.initTmp()
  return m
}
func NullDenseReal64Matrix(rows, cols int) *DenseReal64Matrix {
  m := DenseReal64Matrix{}
  m.values = NullDenseReal64Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  m.initTmp()
  return &m
}
func nilDenseReal64Matrix(rows, cols int) *DenseReal64Matrix {
  m := DenseReal64Matrix{}
  m.values = nilDenseReal64Vector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
}
func AsDenseReal64Matrix(matrix ConstMatrix) *DenseReal64Matrix {
  switch matrix_ := matrix.(type) {
  case *DenseReal64Matrix:
    return matrix_.Clone()
  }
  n, m := matrix.Dims()
  r := NullDenseReal64Matrix(n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i,j).Set(matrix.ConstAt(i,j))
    }
  }
  return r
}
func (matrix *DenseReal64Matrix) initTmp() {
  if len(matrix.tmp1) < matrix.rows {
    matrix.tmp1 = NullDenseReal64Vector(matrix.rows)
  } else {
    matrix.tmp1 = matrix.tmp1[0:matrix.rows]
  }
  if len(matrix.tmp2) < matrix.cols {
    matrix.tmp2 = NullDenseReal64Vector(matrix.cols)
  } else {
    matrix.tmp2 = matrix.tmp2[0:matrix.cols]
  }
}
/* cloning
 * -------------------------------------------------------------------------- */
// Clone matrix including data.
func (matrix *DenseReal64Matrix) Clone() *DenseReal64Matrix {
  return &DenseReal64Matrix{
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
func (matrix *DenseReal64Matrix) index(i, j int) int {
  if i < 0 || j < 0 || i >= matrix.rows || j >= matrix.cols {
    panic(fmt.Errorf("index (%d,%d) out of bounds for matrix of dimension %dx%d", i, j, matrix.rows, matrix.cols))
  }
  if matrix.transposed {
    return (matrix.colOffset + j)*matrix.rowMax + (matrix.rowOffset + i)
  } else {
    return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
  }
}
func (matrix *DenseReal64Matrix) ij(k int) (int, int) {
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
func (matrix *DenseReal64Matrix) AT(i, j int) *Real64 {
  return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseReal64Matrix) ROW(i int) DenseReal64Vector {
  v := nilDenseReal64Vector(matrix.cols)
  for j := 0; j < matrix.cols; j++ {
    v[j] = matrix.values[matrix.index(i, j)].Clone()
  }
  return v
}
func (matrix *DenseReal64Matrix) COL(j int) DenseReal64Vector {
  v := nilDenseReal64Vector(matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    v[i] = matrix.values[matrix.index(i, j)].Clone()
  }
  return v
}
func (matrix *DenseReal64Matrix) DIAG() DenseReal64Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := nilDenseReal64Vector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)].Clone()
  }
  return v
}
func (matrix *DenseReal64Matrix) SLICE(rfrom, rto, cfrom, cto int) *DenseReal64Matrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows = rto - rfrom
  m.colOffset += cfrom
  m.cols = cto - cfrom
  // crop tmp vectors
  m.initTmp()
  return &m
}
func (matrix *DenseReal64Matrix) AsDenseReal64Vector() DenseReal64Vector {
  if matrix.cols < matrix.colMax - matrix.colOffset ||
    (matrix.rows < matrix.rowMax - matrix.rowOffset) {
    n, m := matrix.Dims()
    v := nilDenseReal64Vector(n*m)
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        v[i*matrix.cols + j] = matrix.AT(i, j)
      }
    }
    return v
  } else {
    return DenseReal64Vector(matrix.values)
  }
}
/* matrix interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal64Matrix) CloneMatrix() Matrix {
  return matrix.Clone()
}
func (matrix *DenseReal64Matrix) At(i, j int) Scalar {
  return matrix.AT(i, j)
}
func (a *DenseReal64Matrix) Set(b ConstMatrix) {
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
func (matrix *DenseReal64Matrix) SetIdentity() {
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
func (matrix *DenseReal64Matrix) Reset() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].Reset()
  }
}
func (matrix *DenseReal64Matrix) Row(i int) Vector {
  return matrix.ROW(i)
}
func (matrix *DenseReal64Matrix) Col(j int) Vector {
  return matrix.COL(j)
}
func (matrix *DenseReal64Matrix) Diag() Vector {
  return matrix.DIAG()
}
func (matrix *DenseReal64Matrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal64Matrix) Swap(i1, j1, i2, j2 int) {
  k1 := matrix.index(i1, j1)
  k2 := matrix.index(i2, j2)
  matrix.values[k1], matrix.values[k2] = matrix.values[k2], matrix.values[k1]
}
func (matrix *DenseReal64Matrix) T() Matrix {
  return &DenseReal64Matrix{
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
func (matrix *DenseReal64Matrix) Tip() {
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
func (matrix *DenseReal64Matrix) AsVector() Vector {
  return matrix.AsDenseReal64Vector()
}
func (matrix *DenseReal64Matrix) storageLocation() uintptr {
  return uintptr(unsafe.Pointer(&matrix.values[0]))
}
/* const interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal64Matrix) CloneConstMatrix() ConstMatrix {
  return matrix.Clone()
}
func (matrix *DenseReal64Matrix) Dims() (int, int) {
  if matrix == nil {
    return 0, 0
  } else {
    return matrix.rows, matrix.cols
  }
}
func (matrix *DenseReal64Matrix) Int8At(i, j int) int8 {
  return matrix.values[matrix.index(i, j)].GetInt8()
}
func (matrix *DenseReal64Matrix) Int16At(i, j int) int16 {
  return matrix.values[matrix.index(i, j)].GetInt16()
}
func (matrix *DenseReal64Matrix) Int32At(i, j int) int32 {
  return matrix.values[matrix.index(i, j)].GetInt32()
}
func (matrix *DenseReal64Matrix) Int64At(i, j int) int64 {
  return matrix.values[matrix.index(i, j)].GetInt64()
}
func (matrix *DenseReal64Matrix) IntAt(i, j int) int {
  return matrix.values[matrix.index(i, j)].GetInt()
}
func (matrix *DenseReal64Matrix) Float32At(i, j int) float32 {
  return matrix.values[matrix.index(i, j)].GetFloat32()
}
func (matrix *DenseReal64Matrix) Float64At(i, j int) float64 {
  return matrix.values[matrix.index(i, j)].GetFloat64()
}
func (matrix *DenseReal64Matrix) ConstAt(i, j int) ConstScalar {
  return matrix.values[matrix.index(i, j)]
}
func (matrix *DenseReal64Matrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal64Matrix) ConstRow(i int) ConstVector {
  // no cloning required...
  var v DenseReal64Vector
  if matrix.transposed {
    v = nilDenseReal64Vector(matrix.cols)
    for j := 0; j < matrix.cols; j++ {
      v[j] = matrix.values[matrix.index(i, j)]
    }
  } else {
    i = matrix.index(i, 0)
    v = matrix.values[i:i + matrix.cols]
  }
  return v
}
func (matrix *DenseReal64Matrix) ConstCol(j int) ConstVector {
  // no cloning required...
  var v DenseReal64Vector
  if matrix.transposed {
    j = matrix.index(0, j)
    v = matrix.values[j:j + matrix.rows]
  } else {
    v = nilDenseReal64Vector(matrix.rows)
    for i := 0; i < matrix.rows; i++ {
      v[i] = matrix.values[matrix.index(i, j)]
    }
  }
  return v
}
func (matrix *DenseReal64Matrix) ConstDiag() ConstVector {
  // no cloning required...
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := nilDenseReal64Vector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)]
  }
  return v
}
func (matrix *DenseReal64Matrix) IsSymmetric(epsilon float64) bool {
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
func (matrix *DenseReal64Matrix) AsConstVector() ConstVector {
  return matrix.AsDenseReal64Vector()
}
/* magic interface
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal64Matrix) CloneMagicMatrix() MagicMatrix {
  return matrix.Clone()
}
func (matrix *DenseReal64Matrix) MagicAt(i, j int) MagicScalar {
  return matrix.AT(i, j)
}
func (matrix *DenseReal64Matrix) MagicSlice(rfrom, rto, cfrom, cto int) MagicMatrix {
  return matrix.SLICE(rfrom, rto, cfrom, cto)
}
func (matrix *DenseReal64Matrix) ResetDerivatives() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].ResetDerivatives()
  }
}
func (matrix *DenseReal64Matrix) AsMagicVector() MagicVector {
  return matrix.AsDenseReal64Vector()
}
/* implement MagicScalarContainer
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal64Matrix) Map(f func(Scalar)) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      f(matrix.At(i, j))
    }
  }
}
func (matrix *DenseReal64Matrix) MapSet(f func(ConstScalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.ConstAt(i, j)))
    }
  }
}
func (matrix *DenseReal64Matrix) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.ConstAt(i, j))
    }
  }
  return r
}
func (matrix *DenseReal64Matrix) ElementType() ScalarType {
  return Real64Type
}
func (matrix *DenseReal64Matrix) Variables(order int) error {
  for i, _ := range matrix.values {
    if err := matrix.values[i].SetVariable(i, len(matrix.values), order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (matrix *DenseReal64Matrix) SwapRows(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < m; k++ {
    matrix.Swap(i, k, j, k)
  }
  return nil
}
func (matrix *DenseReal64Matrix) SwapColumns(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return fmt.Errorf("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < n; k++ {
    matrix.Swap(k, i, k, j)
  }
  return nil
}
func (matrix *DenseReal64Matrix) PermuteRows(pi []int) error {
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
func (matrix *DenseReal64Matrix) PermuteColumns(pi []int) error {
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
func (matrix *DenseReal64Matrix) SymmetricPermutation(pi []int) error {
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
func (m *DenseReal64Matrix) String() string {
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
func (a *DenseReal64Matrix) Table() string {
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
func (m *DenseReal64Matrix) Export(filename string) error {
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
func (m *DenseReal64Matrix) Import(filename string) error {
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
      values = append(values, float64(value))
    }
    rows++
  }
  *m = *NewDenseReal64Matrix(values, rows, cols)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *DenseReal64Matrix) MarshalJSON() ([]byte, error) {
  if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
    n, m := obj.Dims()
    tmp := NullDenseReal64Matrix(n, m)
    tmp.Set(obj)
    obj = tmp
  }
  r := struct{Values []*Real64; Rows int; Cols int}{}
  r.Values = obj.values
  r.Rows = obj.rows
  r.Cols = obj.cols
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseReal64Matrix) UnmarshalJSON(data []byte) error {
  r := struct{Values []*Real64; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  obj.values = nilDenseReal64Vector(len(r.Values))
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
func (obj *DenseReal64Matrix) ConstIterator() MatrixConstIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal64Matrix) ConstIteratorFrom(i, j int) MatrixConstIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal64Matrix) MagicIterator() MatrixMagicIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal64Matrix) MagicIteratorFrom(i, j int) MatrixMagicIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal64Matrix) Iterator() MatrixIterator {
  return obj.ITERATOR()
}
func (obj *DenseReal64Matrix) IteratorFrom(i, j int) MatrixIterator {
  return obj.ITERATOR_FROM(i, j)
}
func (obj *DenseReal64Matrix) JointIterator(b ConstMatrix) MatrixJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *DenseReal64Matrix) ITERATOR() *DenseReal64MatrixIterator {
  r := DenseReal64MatrixIterator{obj, 0, -1}
  r.Next()
  return &r
}
func (obj *DenseReal64Matrix) ITERATOR_FROM(i, j int) *DenseReal64MatrixIterator {
  r := DenseReal64MatrixIterator{obj, i, j-1}
  r.Next()
  return &r
}
func (obj *DenseReal64Matrix) JOINT_ITERATOR(b ConstMatrix) *DenseReal64MatrixJointIterator {
  r := DenseReal64MatrixJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, -1, nil, nil}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type DenseReal64MatrixIterator struct {
  m *DenseReal64Matrix
  i, j int
}
func (obj *DenseReal64MatrixIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseReal64MatrixIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseReal64MatrixIterator) GetMagic() MagicScalar {
  return obj.GET()
}
func (obj *DenseReal64MatrixIterator) GET() *Real64 {
  return obj.m.AT(obj.i, obj.j)
}
func (obj *DenseReal64MatrixIterator) Ok() bool {
  return obj.i < obj.m.rowMax && obj.j < obj.m.colMax
}
func (obj *DenseReal64MatrixIterator) next() {
  if obj.j == obj.m.cols-1 {
    obj.i = obj.i + 1
    obj.j = 0
  } else {
    obj.j = obj.j + 1
  }
}
func (obj *DenseReal64MatrixIterator) Next() {
  obj.next()
  for obj.Ok() && obj.GET().nullScalar() {
    obj.next()
  }
}
func (obj *DenseReal64MatrixIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseReal64MatrixIterator) Clone() *DenseReal64MatrixIterator {
  return &DenseReal64MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal64MatrixIterator) CloneIterator() MatrixIterator {
  return &DenseReal64MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal64MatrixIterator) CloneConstIterator() MatrixConstIterator {
  return &DenseReal64MatrixIterator{obj.m, obj.i, obj.j}
}
func (obj *DenseReal64MatrixIterator) CloneMagicIterator() MatrixMagicIterator {
  return &DenseReal64MatrixIterator{obj.m, obj.i, obj.j}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal64MatrixJointIterator struct {
  it1 *DenseReal64MatrixIterator
  it2 MatrixConstIterator
  i, j int
  s1 *Real64
  s2 ConstScalar
}
func (obj *DenseReal64MatrixJointIterator) Index() (int, int) {
  return obj.i, obj.j
}
func (obj *DenseReal64MatrixJointIterator) Ok() bool {
  return !(obj.s1 == nil || obj.s1.GetFloat64() == float64(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == float64(0))
}
func (obj *DenseReal64MatrixJointIterator) Next() {
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
    obj.s2 = ConstFloat64(0.0)
  }
}
func (obj *DenseReal64MatrixJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal64MatrixJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal64MatrixJointIterator) GET() (*Real64, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseReal64MatrixJointIterator) Clone() *DenseReal64MatrixJointIterator {
  r := DenseReal64MatrixJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.i = obj.i
  r.j = obj.j
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseReal64MatrixJointIterator) CloneJointIterator() MatrixJointIterator {
  return obj.Clone()
}
func (obj *DenseReal64MatrixJointIterator) CloneConstJointIterator() MatrixConstJointIterator {
  return obj.Clone()
}
