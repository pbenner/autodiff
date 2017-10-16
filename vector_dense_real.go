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
import "fmt"
import "bytes"
import "bufio"
import "encoding/json"
import "errors"
import "compress/gzip"
import "reflect"
import "sort"
import "strconv"
import "strings"
import "os"
/* vector type declaration
 * -------------------------------------------------------------------------- */
type DenseRealVector []Real
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a vector for scalars of type t (i.e. RealType, or ProbabilityType).
func NewDenseRealVector(values []float64) DenseRealVector {
  v := NilDenseRealVector(len(values))
  for i, _ := range values {
    v[i] = *NewReal(values[i])
  }
  return v
}
// Allocate an empty vector of type t. All values are initialized to zero.
func NullDenseRealVector(length int) DenseRealVector {
  v := NilDenseRealVector(length)
  if length > 0 {
    for i := 0; i < length; i++ {
      v[i] = *NewReal(0.0)
    }
  }
  return v
}
// Create a vector without allocating memory for the scalar variables.
func NilDenseRealVector(length int) DenseRealVector {
  return make(DenseRealVector, length)
}
/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseRealVector) Clone() DenseRealVector {
  result := make(DenseRealVector, len(v))
  for i, _ := range v {
    result[i] = *v[i].Clone()
  }
  return result
}
func (v DenseRealVector) CloneVector() Vector {
  return v.Clone()
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseRealVector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].Set(w.ConstAt(i))
  }
}
/* -------------------------------------------------------------------------- */
func (v DenseRealVector) Dim() int {
  return len(v)
}
func (v DenseRealVector) At(i int) Scalar {
  return &v[i]
}
func (v DenseRealVector) AT(i int) *Real {
  return &v[i]
}
func (v DenseRealVector) ConstAt(i int) ConstScalar {
  return &v[i]
}
func (v DenseRealVector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i].Reset()
  }
}
func (v DenseRealVector) ResetDerivatives() {
  for i := 0; i < len(v); i++ {
    v[i].ResetDerivatives()
  }
}
func (v DenseRealVector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseRealVector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseRealVector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
func (v DenseRealVector) Append(a ...Scalar) Vector {
  for _, s := range a {
    v = append(v, *NewReal(s.GetValue()))
  }
  return v
}
func (v DenseRealVector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(&v[i])
  }
}
func (v DenseRealVector) MapSet(f func(Scalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i].Set(f(&v[i]))
  }
}
func (v DenseRealVector) Reduce(f func(Scalar, Scalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, &v[i])
  }
  return r
}
func (v DenseRealVector) ElementType() ScalarType {
  if len(v) > 0 {
    return reflect.TypeOf(&v[0])
  }
  return nil
}
func (v DenseRealVector) Variables(order int) {
  for i, _ := range v {
    v[i].SetVariable(i, len(v), order)
  }
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) Permute(pi []int) error {
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
type sortDenseRealVectorByValue DenseRealVector
func (v sortDenseRealVectorByValue) Len() int { return len(v) }
func (v sortDenseRealVectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseRealVectorByValue) Less(i, j int) bool { return v[i].GetValue() < v[j].GetValue() }
func (v DenseRealVector) Sort(reverse bool) DenseRealVector {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseRealVectorByValue(v)))
  } else {
    sort.Sort(sortDenseRealVectorByValue(v))
  }
  return v
}
func (v DenseRealVector) SortVector(reverse bool) Vector {
  return v.Sort(reverse)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) ToDenseVector() DenseVector {
  r := NilDenseVector(v.Dim())
  for i := 0; i < v.Dim(); i++ {
    r.SetReferenceAt(i, &v[i])
  }
  return r
}
func (v DenseRealVector) ToMatrix(n, m int) Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseRealMatrix{}
  matrix.values = v
  matrix.rows = n
  matrix.cols = m
  matrix.rowOffset = 0
  matrix.rowMax = n
  matrix.colOffset = 0
  matrix.colMax = m
  matrix.initTmp()
  return &matrix
}
func (v DenseRealVector) SliceFloat64() []float64 {
  s := make([]float64, len(v))
  for i, _ := range v {
    s[i] = v[i].GetValue()
  }
  return s
}
func (v DenseRealVector) String() string {
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
func (v DenseRealVector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v[i].String())
  }
  return buffer.String()
}
func (v DenseRealVector) Export(filename string) error {
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
func (v *DenseRealVector) Import(filename string) error {
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
  // reset vector
  *v = DenseRealVector{}
  for scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    if len(fields) == 0 {
      continue
    }
    if len(*v) != 0 {
      return fmt.Errorf("invalid table")
    }
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return fmt.Errorf("invalid table")
      }
      *v = append(*v, *NewReal(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj DenseRealVector) MarshalJSON() ([]byte, error) {
  r := []Real{}
  r = obj
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseRealVector) UnmarshalJSON(data []byte) error {
  r := []Real{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *obj = NilDenseRealVector(len(r))
  for i := 0; i < len(r); i++ {
    (*obj)[i] = r[i]
  }
  return nil
}
