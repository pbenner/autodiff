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
import "bufio"
import "bytes"
import "compress/gzip"
import "encoding/json"
import "io"
import "os"
import "sort"
import "strconv"
import "strings"
/* -------------------------------------------------------------------------- */
type DenseFloat64Vector []float64
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseFloat64Vector(values []float64) DenseFloat64Vector {
  return DenseFloat64Vector(values)
}
func NullDenseFloat64Vector(n int) DenseFloat64Vector {
  return DenseFloat64Vector(make([]float64, n))
}
// Convert vector type.
func AsDenseFloat64Vector(v ConstVector) DenseFloat64Vector {
  switch v_ := v.(type) {
  case DenseFloat64Vector:
    return v_.Clone()
  }
  values := make([]float64, v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    values[it.Index()] = it.GetConst().GetFloat64()
  }
  return NewDenseFloat64Vector(values)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) Clone() DenseFloat64Vector {
  r := make([]float64, v.Dim())
  copy(r, v)
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) AT(i int) Float64 {
  return Float64{&v[i]}
}
func (v DenseFloat64Vector) APPEND(w DenseFloat64Vector) DenseFloat64Vector {
  return append(v, w...)
}
func (v DenseFloat64Vector) ToDenseFloat64Matrix(n, m int) *DenseFloat64Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseFloat64Matrix{}
  matrix.values = v
  matrix.rows = n
  matrix.cols = m
  matrix.rowOffset = 0
  matrix.rowMax = n
  matrix.colOffset = 0
  matrix.colMax = m
  return &matrix
}
/* vector interface
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseFloat64Vector) At(i int) Scalar {
  return v.AT(i)
}
func (v DenseFloat64Vector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i] = w.ConstAt(i).GetFloat64()
  }
}
func (v DenseFloat64Vector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i] = 0.0
  }
}
func (v DenseFloat64Vector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseFloat64Vector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseFloat64Vector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseFloat64Vector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    v = append(v, scalar.GetFloat64())
  }
  return v
}
func (v DenseFloat64Vector) AppendVector(w Vector) Vector {
  for i := 0; i < w.Dim(); i++ {
    v = append(v, w.ConstAt(i).GetFloat64())
  }
  return v
}
func (v DenseFloat64Vector) AsMatrix(n, m int) Matrix {
  return v.ToDenseFloat64Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseFloat64Vector) Dim() int {
  return len(v)
}
func (v DenseFloat64Vector) Int8At(i int) int8 {
  return int8(v[i])
}
func (v DenseFloat64Vector) Int16At(i int) int16 {
  return int16(v[i])
}
func (v DenseFloat64Vector) Int32At(i int) int32 {
  return int32(v[i])
}
func (v DenseFloat64Vector) Int64At(i int) int64 {
  return int64(v[i])
}
func (v DenseFloat64Vector) IntAt(i int) int {
  return int(v[i])
}
func (v DenseFloat64Vector) Float32At(i int) float32 {
  return float32(v[i])
}
func (v DenseFloat64Vector) Float64At(i int) float64 {
  return float64(v[i])
}
func (v DenseFloat64Vector) ConstAt(i int) ConstScalar {
  return Float64{&v[i]}
}
func (v DenseFloat64Vector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(v.At(i))
  }
}
func (v DenseFloat64Vector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i] = f(v.ConstAt(i)).GetFloat64()
  }
}
func (v DenseFloat64Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseFloat64Vector) ElementType() ScalarType {
  return Float64Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) Permute(pi []int) error {
  if len(pi) != len(v) {
    return fmt.Errorf("Permute(): permutation vector has invalid length!")
  }
  // permute vector
  for i := 0; i < len(v); i++ {
    if pi[i] < 0 || pi[i] >= len(v) {
      return fmt.Errorf("Permute(): invalid permutation")
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
type sortDenseFloat64VectorByValue DenseFloat64Vector
func (v sortDenseFloat64VectorByValue) Len() int { return len(v) }
func (v sortDenseFloat64VectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseFloat64VectorByValue) Less(i, j int) bool { return v[i] < v[j] }
func (v DenseFloat64Vector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseFloat64VectorByValue(v)))
  } else {
    sort.Sort(sortDenseFloat64VectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) String() string {
  var buffer bytes.Buffer
  buffer.WriteString("[")
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(v.ConstAt(i).String())
  }
  buffer.WriteString("]")
  return buffer.String()
}
func (v DenseFloat64Vector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    buffer.WriteString(v.ConstAt(i).String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (v DenseFloat64Vector) Export(filename string) error {
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
func (v *DenseFloat64Vector) Import(filename string) error {
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
  // reset vector
  *v = DenseFloat64Vector{}
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
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return fmt.Errorf("invalid table")
      }
      *v = append(*v, float64(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) MarshalJSON() ([]byte, error) {
  r := []float64{}
  r = v
  return json.MarshalIndent(r, "", "  ")
}
func (v *DenseFloat64Vector) UnmarshalJSON(data []byte) error {
  r := []float64{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *v = r
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (v DenseFloat64Vector) ConstIterator() VectorConstIterator {
  return v.ITERATOR()
}
func (v DenseFloat64Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseFloat64Vector) Iterator() VectorIterator {
  return v.ITERATOR()
}
func (v DenseFloat64Vector) IteratorFrom(i int) VectorIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseFloat64Vector) JointIterator(b ConstVector) VectorJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseFloat64Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseFloat64Vector) ITERATOR() *DenseFloat64VectorIterator {
  r := DenseFloat64VectorIterator{v, -1}
  r.Next()
  return &r
}
func (v DenseFloat64Vector) ITERATOR_FROM(i int) *DenseFloat64VectorIterator {
  r := DenseFloat64VectorIterator{v, i-1}
  r.Next()
  return &r
}
func (v DenseFloat64Vector) JOINT_ITERATOR(b ConstVector) *DenseFloat64VectorJointIterator {
  r := DenseFloat64VectorJointIterator{}
  r.it1 = v.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* const iterator
 * -------------------------------------------------------------------------- */
type DenseFloat64VectorIterator struct {
  v DenseFloat64Vector
  i int
}
func (obj *DenseFloat64VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseFloat64VectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseFloat64VectorIterator) GET() Float64 {
  return Float64{&obj.v[obj.i]}
}
func (obj *DenseFloat64VectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseFloat64VectorIterator) Next() {
  obj.i++
}
func (obj *DenseFloat64VectorIterator) Index() int {
  return obj.i
}
func (obj *DenseFloat64VectorIterator) Clone() *DenseFloat64VectorIterator {
  return &DenseFloat64VectorIterator{obj.v, obj.i}
}
func (obj *DenseFloat64VectorIterator) CloneIterator() VectorIterator {
  return &DenseFloat64VectorIterator{obj.v, obj.i}
}
func (obj *DenseFloat64VectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseFloat64VectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseFloat64VectorJointIterator struct {
  it1 *DenseFloat64VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Float64
  s2 ConstScalar
}
func (obj *DenseFloat64VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseFloat64VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}
func (obj *DenseFloat64VectorJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1.ptr = nil
  obj.s2 = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GetConst()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1.ptr != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstFloat64(0.0)
  }
}
func (obj *DenseFloat64VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseFloat64VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseFloat64VectorJointIterator) GET() (Float64, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseFloat64VectorJointIterator) Clone() *DenseFloat64VectorJointIterator {
  r := DenseFloat64VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseFloat64VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
func (obj *DenseFloat64VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
