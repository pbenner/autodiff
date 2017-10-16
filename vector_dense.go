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

import "fmt"
import "bytes"
import "bufio"
import "compress/gzip"
import "encoding/json"
import "errors"
import "reflect"
import "sort"
import "strconv"
import "strings"
import "os"

/* vector type declaration
 * -------------------------------------------------------------------------- */

type DenseVector []Scalar

/* constructors
 * -------------------------------------------------------------------------- */

// Allocate a vector for scalars of type t (i.e. RealType, or ProbabilityType).
func NewDenseVector(t ScalarType, values []float64) DenseVector {
  v := NilDenseVector(len(values))
  f := ScalarConstructor(t)
  for i, _ := range values {
    v[i] = f(values[i])
  }
  return v
}

// Allocate an empty vector of type t. All values are initialized to zero.
func NullDenseVector(t ScalarType, length int) DenseVector {
  v := NilDenseVector(length)
  if length > 0 {
    f := ScalarConstructor(t)
    for i := 0; i < length; i++ {
      v[i] = f(0.0)
    }
  }
  return v
}

// Create a vector without allocating memory for the scalar variables.
func NilDenseVector(length int) DenseVector {
  return make(DenseVector, length)
}

/* -------------------------------------------------------------------------- */

// Create a deep copy of the vector.
func (v DenseVector) Clone() DenseVector {
  result := make(DenseVector, len(v))

  for i, _ := range v {
    result[i] = v[i].CloneScalar()
  }
  return result
}

func (v DenseVector) CloneVector() Vector {
  return v.Clone()
}

// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseVector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].Set(w.ConstAt(i))
  }
}

/* -------------------------------------------------------------------------- */

func (v DenseVector) Dim() int {
  return len(v)
}

func (v DenseVector) At(i int) Scalar {
  return v[i]
}

func (v DenseVector) ConstAt(i int) ConstScalar {
  return v[i]
}

func (v DenseVector) RealAt(i int) *Real {
  return v[i].(*Real)
}

func (v DenseVector) BareRealAt(i int) *BareReal {
  return v[i].(*BareReal)
}

func (v DenseVector) SetReferenceAt(i int, s Scalar) {
  v[i] = s
}

func (v DenseVector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i].Reset()
  }
}

func (v DenseVector) ResetDerivatives() {
  for i := 0; i < len(v); i++ {
    v[i].ResetDerivatives()
  }
}

func (v DenseVector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}

func (v DenseVector) Slice(i, j int) Vector {
  return v[i:j]
}

func (v DenseVector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}

func (v DenseVector) Append(a ...Scalar) Vector {
  return append(v, a...)
}

func (v DenseVector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */

func (v DenseVector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(v[i])
  }
}

func (v DenseVector) MapSet(f func(Scalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i] = f(v[i])
  }
}

func (v DenseVector) Reduce(f func(Scalar, Scalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v[i])
  }
  return r
}

func (v DenseVector) ElementType() ScalarType {
  if len(v) > 0 {
    return reflect.TypeOf(v[0])
  }
  return nil
}

func (v DenseVector) ConvertElementType(t ScalarType) {
  for i := 0; i < len(v); i++ {
    v[i] = NewScalar(t, v[i].GetValue())
  }
}

func (v DenseVector) Variables(order int) error {
  return Variables(order, v...)
}

/* permutations
 * -------------------------------------------------------------------------- */

func (v DenseVector) Permute(pi []int) error {
  if len(pi) != len(v) {
    return errors.New("Permute(): permutation vector has invalid length!")
  }
  // permute vector
  for i := 0; i < len(v); i++ {
    if pi[i] < 0 || pi[i] > len(v) {
      return errors.New("SymmetricPermutation(): invalid permutation")
    }
    if i != pi[i] && pi[i] > i {
      // permute elements
      v[pi[i]], v[i] = v[i], v[pi[i]]
    }
  }
  return nil
}

/* sorting
 * -------------------------------------------------------------------------- */

type sortDenseVectorByValue DenseVector

func (v sortDenseVectorByValue) Len() int           { return len(v) }
func (v sortDenseVectorByValue) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v sortDenseVectorByValue) Less(i, j int) bool { return v[i].GetValue() < v[j].GetValue() }

func (v DenseVector) Sort(reverse bool) DenseVector {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseVectorByValue(v)))
  } else {
    sort.Sort(sortDenseVectorByValue(v))
  }
  return v
}

func (v DenseVector) SortVector(reverse bool) Vector {
  return v.Sort(reverse)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (v DenseVector) ToDenseVector() DenseVector {
  return v
}

func (v DenseVector) ToMatrix(n, m int) Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseMatrix{}
  matrix.values    = v
  matrix.rows      = n
  matrix.cols      = m
  matrix.rowOffset = 0
  matrix.rowMax    = n
  matrix.colOffset = 0
  matrix.colMax    = m
  matrix.initTmp()
  return &matrix
}

func (v DenseVector) SliceFloat64() []float64 {
  s := make([]float64, len(v))
  for i, _ := range v {
    s[i] = v[i].GetValue()
  }
  return s
}

func (v DenseVector) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(v[i].String())
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (v DenseVector) Table() string {
  var buffer bytes.Buffer

  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v[i].String())
  }

  return buffer.String()
}

func (v DenseVector) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  if _, err := fmt.Fprintf(w, "%s\n", v.Table()); err != nil {
    return err
  }
  return nil
}

func ImportDenseVector(t ScalarType, filename string) (DenseVector, error) {
  result := NewDenseVector(t, []float64{})

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
    if len(result) != 0 {
      return result, errors.New("invalid table")
    }
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return result, errors.New("invalid table")
      }
      result = append(result, NewScalar(t, value))
    }
  }
  return result, nil
}

/* json
 * -------------------------------------------------------------------------- */

func (obj DenseVector) MarshalJSON() ([]byte, error) {
  r := []Scalar{}
  r  = obj
  return json.MarshalIndent(r, "", "  ")
}

func (obj *DenseVector) UnmarshalJSON(data []byte) error {
  r := []*Real{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *obj = NilDenseVector(len(r))
  for i := 0; i < len(r); i++ {
    (*obj)[i] = r[i]
  }
  return nil
}
