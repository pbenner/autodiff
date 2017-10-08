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

import "bytes"
import "bufio"
import "compress/gzip"
import "encoding/json"
import "errors"
import "fmt"
import "reflect"
import "strconv"
import "strings"
import "os"
import "unsafe"

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type DenseMatrix struct {
  values     DenseVector
  rows       int
  cols       int
  rowOffset  int
  rowMax     int
  colOffset  int
  colMax     int
  transposed bool
  tmp1       DenseVector
  tmp2       DenseVector
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewMatrix(t ScalarType, rows, cols int, values []float64) Matrix {
  return NewDenseMatrix(t, rows, cols, values)
}

func NullMatrix(t ScalarType, rows, cols int) Matrix {
  return NullDenseMatrix(t, rows, cols)
}

func NilMatrix(rows, cols int) Matrix {
  return NilDenseMatrix(rows, cols)
}

func NewDenseMatrix(t ScalarType, rows, cols int, values []float64) *DenseMatrix {
  m := NilDenseMatrix(rows, cols)
  v := m.values
  f := ScalarConstructor(t)
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      v[i] = f(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      v[i] = f(values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  m.initTmp()

  return m
}

func NullDenseMatrix(t ScalarType, rows, cols int) *DenseMatrix {
  m := DenseMatrix{}
  m.values    = NullDenseVector(t, rows*cols)
  m.rows      = rows
  m.cols      = cols
  m.rowOffset = 0
  m.rowMax    = rows
  m.colOffset = 0
  m.colMax    = cols
  m.initTmp()
  return &m
}

func NilDenseMatrix(rows, cols int) *DenseMatrix {
  m := DenseMatrix{}
  m.values    = NilDenseVector(rows*cols)
  m.rows      = rows
  m.cols      = cols
  m.rowOffset = 0
  m.rowMax    = rows
  m.colOffset = 0
  m.colMax    = cols
  return &m
}

func (matrix *DenseMatrix) initTmp() {
  t := matrix.ElementType()
  if len(matrix.tmp1) < matrix.rows {
    matrix.tmp1 = NullDenseVector(t, matrix.rows)
  } else {
    matrix.tmp1 = matrix.tmp1[0:matrix.rows]
  }
  if len(matrix.tmp2) < matrix.cols {
    matrix.tmp2 = NullDenseVector(t, matrix.cols)
  } else {
    matrix.tmp2 = matrix.tmp2[0:matrix.cols]
  }
}

/* cloning
 * -------------------------------------------------------------------------- */

// Clone matrix including data.
func (matrix *DenseMatrix) Clone() *DenseMatrix {
  return &DenseMatrix{
    values    : matrix.values.Clone(),
    rows      : matrix.rows,
    cols      : matrix.cols,
    transposed: matrix.transposed,
    rowOffset : matrix.rowOffset,
    rowMax    : matrix.rowMax,
    colOffset : matrix.colOffset,
    colMax    : matrix.colMax,
    tmp1      : matrix.tmp1.Clone(),
    tmp2      : matrix.tmp2.Clone() }
}

func (matrix *DenseMatrix) CloneMatrix() Matrix {
  return matrix.Clone()
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func IdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.At(i, i).Set(NewScalar(t, 1))
  }
  return matrix
}

/* field access
 * -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) index(i, j int) int {
  if matrix.transposed {
    return (matrix.colOffset + j)*matrix.rowMax + (matrix.rowOffset + i)
  } else {
    return (matrix.rowOffset + i)*matrix.colMax + (matrix.colOffset + j)
  }
}

func (matrix *DenseMatrix) Dims() (int, int) {
  if matrix == nil {
    return 0, 0
  } else {
    return matrix.rows, matrix.cols
  }
}

func (matrix *DenseMatrix) Row(i int) Vector {
  var v DenseVector
  if matrix.transposed {
    v = NilDenseVector(matrix.cols)
    for j := 0; j < matrix.cols; j++ {
      v[j] = matrix.values[matrix.index(i, j)]
    }
  } else {
    i = matrix.index(i, 0)
    v = matrix.values[i:i + matrix.cols]
  }
  return v
}

func (matrix *DenseMatrix) Col(j int) Vector {
  var v DenseVector
  if matrix.transposed {
    j = matrix.index(0, j)
    v = matrix.values[j:j + matrix.rows]
  } else {
    v = NilDenseVector(matrix.rows)
    for i := 0; i < matrix.rows; i++ {
      v[i] = matrix.values[matrix.index(i, j)]
    }
  }
  return v
}

func (matrix *DenseMatrix) Diag() Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := NilDenseVector(n)
  for i := 0; i < n; i++ {
    v.SetReferenceAt(i, matrix.values[matrix.index(i, i)])
  }
  return v
}

func (matrix *DenseMatrix) Slice(rfrom, rto, cfrom, cto int) Matrix {
  m := *matrix
  m.rowOffset += rfrom
  m.rows       = rto - rfrom
  m.colOffset += cfrom
  m.cols       = cto - cfrom
  // crop tmp vectors
  m.initTmp()
  return &m
}

func (matrix *DenseMatrix) Swap(i1, j1, i2, j2 int) {
  k1 := matrix.index(i1, j1)
  k2 := matrix.index(i2, j2)
  matrix.values[k1], matrix.values[k2] = matrix.values[k2], matrix.values[k1]
}

func (matrix *DenseMatrix) ToVector() Vector {
  return matrix.ToDenseVector()
}

func (matrix *DenseMatrix) ToDenseVector() DenseVector {
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
    return matrix.values
  }
}

func (matrix *DenseMatrix) ToDenseMatrix() *DenseMatrix {
  return matrix
}

/* -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) T() Matrix {
  return &DenseMatrix{
    values    :  matrix.values,
    rows      :  matrix.cols,
    cols      :  matrix.rows,
    transposed: !matrix.transposed,
    rowOffset :  matrix.colOffset,
    rowMax    :  matrix.colMax,
    colOffset :  matrix.rowOffset,
    colMax    :  matrix.rowMax,
    tmp1      :  matrix.tmp2,
    tmp2      :  matrix.tmp1 }
}

/* -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) ConstAt(i, j int) ConstScalar {
  return matrix.values[matrix.index(i, j)]
}

func (matrix *DenseMatrix) ConstSlice(rfrom, rto, cfrom, cto int) ConstMatrix {
  return matrix.Slice(rfrom, rto, cfrom, cto)
}

func (matrix *DenseMatrix) ConstRow(i int) ConstVector {
  return matrix.Row(i)
}

func (matrix *DenseMatrix) ConstCol(i int) ConstVector {
  return matrix.Col(i)
}

func (matrix *DenseMatrix) ConstDiag() ConstVector {
  return matrix.Diag()
}

/* -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) At(i, j int) Scalar {
  return matrix.values[matrix.index(i, j)]
}

func (matrix *DenseMatrix) RealAt(i, j int) *Real {
  return matrix.values[matrix.index(i, j)].(*Real)
}

func (matrix *DenseMatrix) BareRealAt(i, j int) *BareReal {
  return matrix.values[matrix.index(i, j)].(*BareReal)
}

func (matrix *DenseMatrix) SetReferenceAt(i, j int, s Scalar) {
  matrix.values[matrix.index(i, j)] = s
}

func (matrix *DenseMatrix) Reset() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].Reset()
  }
}

func (matrix *DenseMatrix) ResetDerivatives() {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i].ResetDerivatives()
  }
}

func (a *DenseMatrix) Set(b Matrix) {
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

func (matrix *DenseMatrix) SetIdentity() {
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

func (matrix *DenseMatrix) IsSymmetric(epsilon float64) bool {
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

func (matrix *DenseMatrix) storageLocation() uintptr {
  return uintptr(unsafe.Pointer(&matrix.values[0]))
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) Map(f func(Scalar)) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      f(matrix.At(i, j))
    }
  }
}

func (matrix *DenseMatrix) MapSet(f func(Scalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      matrix.At(i,j).Set(f(matrix.At(i, j)))
    }
  }
}

func (matrix *DenseMatrix) Reduce(f func(Scalar, Scalar) Scalar, r Scalar) Scalar {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.At(i, j))
    }
  }
  return r
}

func (matrix *DenseMatrix) ElementType() ScalarType {
  if matrix.rows > 0 && matrix.cols > 0 {
    return reflect.TypeOf(matrix.values[0])
  }
  return nil
}

func (matrix *DenseMatrix) ConvertElementType(t ScalarType) {
  for i := 0; i < len(matrix.values); i++ {
    matrix.values[i] = NewScalar(t, matrix.values[i].GetValue())
  }
  matrix.tmp1.ConvertElementType(t)
  matrix.tmp2.ConvertElementType(t)
}

func (matrix *DenseMatrix) Variables(order int) {
  Variables(order, matrix.values...)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (m *DenseMatrix) String() string {
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

func (a *DenseMatrix) Table() string {
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

func (m *DenseMatrix) Export(filename string) error {
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

func ReadMatrix(t ScalarType, filename string) (Matrix, error) {
  result := NewMatrix(t, 0, 0, []float64{})
  data   := []float64{}
  rows   := 0
  cols   := 0

  var scanner *bufio.Scanner
  // open file
  f, err := os.Open(filename)
  if err != nil {
    return result, err
  }
  defer f.Close()
  isgzip, err := isGzip(filename)
  if err != nil {
    return result, err
  }
  // check if file is gzipped
  if isgzip {
    g, err := gzip.NewReader(f)
    if err != nil {
      return result, err
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
      return result, errors.New("invalid table")
    }
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return result, errors.New("invalid table")
      }
      data = append(data, value)
    }
    rows++
  }
  result = NewMatrix(t, rows, cols, data)

  return result, nil
}

/* json
 * -------------------------------------------------------------------------- */

func (obj *DenseMatrix) MarshalJSON() ([]byte, error) {
  if obj.transposed || obj.rowMax > obj.rows || obj.colMax > obj.cols {
    n, m := obj.Dims()
    tmp  := NullDenseMatrix(obj.ElementType(), n, m)
    tmp.Set(obj)
    obj = tmp
  }
  r := struct{Values DenseVector; Rows int; Cols int}{}
  r.Values = obj.values
  r.Rows   = obj.rows
  r.Cols   = obj.cols
  return json.MarshalIndent(r, "", "  ")
}

func (obj *DenseMatrix) UnmarshalJSON(data []byte) error {
  r := struct{Values []*Real; Rows int; Cols int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  obj.values = NilDenseVector(len(r.Values))
  for i := 0; i < len(r.Values); i++ {
    obj.values[i] = r.Values[i]
  }
  obj.rows       = r.Rows
  obj.rowMax     = r.Rows
  obj.rowOffset  = 0
  obj.cols       = r.Cols
  obj.colMax     = r.Cols
  obj.colOffset  = 0
  obj.transposed = false
  obj.initTmp()
  return nil
}
