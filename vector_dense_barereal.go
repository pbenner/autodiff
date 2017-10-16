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
type DenseBareRealVector []BareReal
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a vector for scalars of type t (i.e. RealType, or ProbabilityType).
func NewDenseBareRealVector(values []float64) DenseBareRealVector {
  v := NilDenseBareRealVector(len(values))
  for i, _ := range values {
    v[i] = *NewBareReal(values[i])
  }
  return v
}
// Allocate an empty vector of type t. All values are initialized to zero.
func NullDenseBareRealVector(length int) DenseBareRealVector {
  v := NilDenseBareRealVector(length)
  if length > 0 {
    for i := 0; i < length; i++ {
      v[i] = *NewBareReal(0.0)
    }
  }
  return v
}
// Create a vector without allocating memory for the scalar variables.
func NilDenseBareRealVector(length int) DenseBareRealVector {
  return make(DenseBareRealVector, length)
}
/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseBareRealVector) Clone() DenseBareRealVector {
  result := make(DenseBareRealVector, len(v))
  for i, _ := range v {
    result[i] = *v[i].Clone()
  }
  return result
}
func (v DenseBareRealVector) CloneVector() Vector {
  return v.Clone()
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseBareRealVector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].Set(w.ConstAt(i))
  }
}
/* -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Dim() int {
  return len(v)
}
func (v DenseBareRealVector) At(i int) Scalar {
  return &v[i]
}
func (v DenseBareRealVector) AT(i int) *BareReal {
  return &v[i]
}
func (v DenseBareRealVector) ConstAt(i int) ConstScalar {
  return &v[i]
}
func (v DenseBareRealVector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i].Reset()
  }
}
func (v DenseBareRealVector) ResetDerivatives() {
  for i := 0; i < len(v); i++ {
    v[i].ResetDerivatives()
  }
}
func (v DenseBareRealVector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseBareRealVector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseBareRealVector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
func (v DenseBareRealVector) Append(a ...Scalar) Vector {
  for _, s := range a {
    v = append(v, *NewBareReal(s.GetValue()))
  }
  return v
}
func (v DenseBareRealVector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(&v[i])
  }
}
func (v DenseBareRealVector) MapSet(f func(Scalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i].Set(f(&v[i]))
  }
}
func (v DenseBareRealVector) Reduce(f func(Scalar, Scalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, &v[i])
  }
  return r
}
func (v DenseBareRealVector) ElementType() ScalarType {
  if len(v) > 0 {
    return reflect.TypeOf(&v[0])
  }
  return nil
}
func (v DenseBareRealVector) Variables(order int) {
  for i, _ := range v {
    v[i].SetVariable(i, len(v), order)
  }
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Permute(pi []int) error {
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
type sortDenseBareRealVectorByValue DenseBareRealVector
func (v sortDenseBareRealVectorByValue) Len() int { return len(v) }
func (v sortDenseBareRealVectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseBareRealVectorByValue) Less(i, j int) bool { return v[i].GetValue() < v[j].GetValue() }
func (v DenseBareRealVector) Sort(reverse bool) DenseBareRealVector {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseBareRealVectorByValue(v)))
  } else {
    sort.Sort(sortDenseBareRealVectorByValue(v))
  }
  return v
}
func (v DenseBareRealVector) SortVector(reverse bool) Vector {
  return v.Sort(reverse)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) ToDenseVector() DenseVector {
  r := NilDenseVector(v.Dim())
  for i := 0; i < v.Dim(); i++ {
    r.SetReferenceAt(i, &v[i])
  }
  return r
}
func (v DenseBareRealVector) ToMatrix(n, m int) Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseBareRealMatrix{}
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
func (v DenseBareRealVector) SliceFloat64() []float64 {
  s := make([]float64, len(v))
  for i, _ := range v {
    s[i] = v[i].GetValue()
  }
  return s
}
func (v DenseBareRealVector) String() string {
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
func (v DenseBareRealVector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v[i].String())
  }
  return buffer.String()
}
func (v DenseBareRealVector) Export(filename string) error {
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
func (v *DenseBareRealVector) Import(filename string) error {
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
  *v = DenseBareRealVector{}
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
      *v = append(*v, *NewBareReal(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj DenseBareRealVector) MarshalJSON() ([]byte, error) {
  r := []BareReal{}
  r = obj
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseBareRealVector) UnmarshalJSON(data []byte) error {
  r := []BareReal{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *obj = NilDenseBareRealVector(len(r))
  for i := 0; i < len(r); i++ {
    (*obj)[i] = r[i]
  }
  return nil
}
