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
package autodiff
/* -------------------------------------------------------------------------- */
import "bytes"
import "bufio"
import "encoding/json"
import "fmt"
import "reflect"
import "os"
import "unsafe"
/* matrix type declaration
 * -------------------------------------------------------------------------- */
type DenseRealMatrix struct {
  values DenseRealVector
  rows int
  cols int
  rowOffset int
  rowMax int
  colOffset int
  colMax int
  transposed bool
  tmp1 DenseRealVector
  tmp2 DenseRealVector
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseRealMatrix(rows, cols int, values []float64) *DenseRealMatrix {
  m := NilDenseRealMatrix(rows, cols)
  v := m.values
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      v[i] = *NewReal(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      v[i] = *NewReal(values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  m.initTmp()
  return m
}
func NullDenseRealMatrix(rows, cols int) *DenseRealMatrix {
  m := DenseRealMatrix{}
  m.values = NullDenseRealVector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  m.initTmp()
  return &m
}
func NilDenseRealMatrix(rows, cols int) *DenseRealMatrix {
  m := DenseRealMatrix{}
  m.values = NilDenseRealVector(rows*cols)
  m.rows = rows
  m.cols = cols
  m.rowOffset = 0
  m.rowMax = rows
  m.colOffset = 0
  m.colMax = cols
  return &m
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
func (matrix *DenseRealMatrix) CloneMatrix() Matrix {
  return matrix.Clone()
}
/* field access
 * -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) index(i, j int) int {
  if matrix.transposed {
    return (matrix.colOffset + j)*matrix.rowMax + (matrix.rowOffset + i)
  } else {
    return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
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
  var v DenseRealVector
  if matrix.transposed {
    v = NilDenseRealVector(matrix.cols)
    for j := 0; j < matrix.cols; j++ {
      v[j] = matrix.values[matrix.index(i, j)]
    }
  } else {
    i = matrix.index(i, 0)
    v = matrix.values[i:i + matrix.cols]
  }
  return v
}
func (matrix *DenseRealMatrix) Col(j int) Vector {
  var v DenseRealVector
  if matrix.transposed {
    j = matrix.index(0, j)
    v = matrix.values[j:j + matrix.rows]
  } else {
    v = NilDenseRealVector(matrix.rows)
    for i := 0; i < matrix.rows; i++ {
      v[i] = matrix.values[matrix.index(i, j)]
    }
  }
  return v
}
func (matrix *DenseRealMatrix) Diag() Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := NilDenseVector(n)
  for i := 0; i < n; i++ {
    v.SetReferenceAt(i, &matrix.values[matrix.index(i, i)])
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
func (matrix *DenseRealMatrix) ToVector() Vector {
  return matrix.ToDenseVector()
}
func (matrix *DenseRealMatrix) ToDenseVector() DenseVector {
  if matrix.cols < matrix.colMax - matrix.colOffset ||
    (matrix.rows < matrix.rowMax - matrix.rowOffset) {
    n, m := matrix.Dims()
    v := NilDenseVector(n*m)
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        v[i*matrix.cols + j] = matrix.At(i, j)
      }
    }
    return v
  } else {
    return DenseRealVector(matrix.values).ToDenseVector()
  }
}
func (matrix *DenseRealMatrix) ToDenseMatrix() *DenseMatrix {
  r := DenseMatrix{}
  r.values = matrix.values.ToDenseVector()
  r.rows = matrix.rows
  r.cols = matrix.cols
  r.rowOffset = matrix.rowOffset
  r.rowMax = matrix.rowMax
  r.colOffset = matrix.colOffset
  r.colMax = matrix.colMax
  r.initTmp()
  return &r
}
/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) T() Matrix {
  return &DenseRealMatrix{
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
/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) ConstAt(i, j int) ConstScalar {
  return &matrix.values[matrix.index(i, j)]
}
func (matrix *DenseRealMatrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  return matrix.Slice(rfrom, rto, cfrom, cto)
}
func (matrix *DenseRealMatrix) ConstRow(i int) ConstVector {
  return matrix.Row(i)
}
func (matrix *DenseRealMatrix) ConstCol(i int) ConstVector {
  return matrix.Col(i)
}
func (matrix *DenseRealMatrix) ConstDiag() ConstVector {
  return matrix.Diag()
}
/* -------------------------------------------------------------------------- */
func (matrix *DenseRealMatrix) At(i, j int) Scalar {
  return &matrix.values[matrix.index(i, j)]
}
func (matrix *DenseRealMatrix) AT(i, j int) *Real {
  return &matrix.values[matrix.index(i, j)]
}
func (matrix *DenseRealMatrix) Reset() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].Reset()
  }
}
func (matrix *DenseRealMatrix) ResetDerivatives() {
}
func (a *DenseRealMatrix) Set(b Matrix) {
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("Copy(): Matrix dimension does not match!")
  }
  for i := 0; i < n1; i++ {
    for j := 0; j < m1; j++ {
      a.At(i, j).Set(b.At(i, j).CloneScalar())
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
    for j := i+1; j < m; j++ {
      if !matrix.At(i,j).Equals(matrix.At(j,i), 1e-12) {
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
func (matrix *DenseRealMatrix) MapSet(f func(Scalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.At(i, j)))
    }
  }
}
func (matrix *DenseRealMatrix) Reduce(f func(Scalar, Scalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.At(i, j))
    }
  }
  return r
}
func (matrix *DenseRealMatrix) ElementType() ScalarType {
  if matrix.rows > 0 && matrix.cols > 0 {
    return reflect.TypeOf(&matrix.values[0])
  }
  return nil
}
func (matrix *DenseRealMatrix) Variables(order int) {
  for i, _ := range matrix.values {
    matrix.values[i].SetVariable(i, len(matrix.values), order)
  }
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
      buffer.WriteString(m.At(i,j).String())
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
      buffer.WriteString(a.At(i,j).String())
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
/* json
 * -------------------------------------------------------------------------- */
func (obj *DenseRealMatrix) MarshalJSON() ([]byte, error) {
  if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
    n, m := obj.Dims()
    tmp := NullDenseRealMatrix(n, m)
    tmp.Set(obj)
    obj = tmp
  }
  r := struct{Values DenseRealVector; Rows int; Cols int}{}
  r.Values = obj.values
  r.Rows = obj.rows
  r.Cols = obj.cols
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseRealMatrix) UnmarshalJSON(data []byte) error {
  r := struct{Values []Real; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  obj.values = NilDenseRealVector(len(r.Values))
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
