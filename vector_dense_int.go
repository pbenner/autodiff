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
type DenseIntVector []int
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseIntVector(values []int) DenseIntVector {
  return DenseIntVector(values)
}
func NullDenseIntVector(n int) DenseIntVector {
  return DenseIntVector(make([]int, n))
}
// Convert vector type.
func AsDenseIntVector(v ConstVector) DenseIntVector {
  switch v_ := v.(type) {
  case DenseIntVector:
    return v_.Clone()
  }
  values := make([]int, v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    values[it.Index()] = it.GetConst().GetInt()
  }
  return NewDenseIntVector(values)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) Clone() DenseIntVector {
  r := make([]int, v.Dim())
  copy(r, v)
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) AT(i int) Int {
  return Int{&v[i]}
}
func (v DenseIntVector) APPEND(w DenseIntVector) DenseIntVector {
  return append(v, w...)
}
func (v DenseIntVector) ToDenseIntMatrix(n, m int) *DenseIntMatrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseIntMatrix{}
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
func (v DenseIntVector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseIntVector) At(i int) Scalar {
  return v.AT(i)
}
func (v DenseIntVector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i] = w.ConstAt(i).GetInt()
  }
}
func (v DenseIntVector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i] = 0.0
  }
}
func (v DenseIntVector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseIntVector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseIntVector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseIntVector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    v = append(v, scalar.GetInt())
  }
  return v
}
func (v DenseIntVector) AppendVector(w Vector) Vector {
  for i := 0; i < w.Dim(); i++ {
    v = append(v, w.ConstAt(i).GetInt())
  }
  return v
}
func (v DenseIntVector) AsMatrix(n, m int) Matrix {
  return v.ToDenseIntMatrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseIntVector) Dim() int {
  return len(v)
}
func (v DenseIntVector) Int8At(i int) int8 {
  return int8(v[i])
}
func (v DenseIntVector) Int16At(i int) int16 {
  return int16(v[i])
}
func (v DenseIntVector) Int32At(i int) int32 {
  return int32(v[i])
}
func (v DenseIntVector) Int64At(i int) int64 {
  return int64(v[i])
}
func (v DenseIntVector) IntAt(i int) int {
  return int(v[i])
}
func (v DenseIntVector) Float32At(i int) float32 {
  return float32(v[i])
}
func (v DenseIntVector) Float64At(i int) float64 {
  return float64(v[i])
}
func (v DenseIntVector) ConstAt(i int) ConstScalar {
  return Int{&v[i]}
}
func (v DenseIntVector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(v.At(i))
  }
}
func (v DenseIntVector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i] = f(v.ConstAt(i)).GetInt()
  }
}
func (v DenseIntVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseIntVector) ElementType() ScalarType {
  return IntType
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) Permute(pi []int) error {
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
type sortDenseIntVectorByValue DenseIntVector
func (v sortDenseIntVectorByValue) Len() int { return len(v) }
func (v sortDenseIntVectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseIntVectorByValue) Less(i, j int) bool { return v[i] < v[j] }
func (v DenseIntVector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseIntVectorByValue(v)))
  } else {
    sort.Sort(sortDenseIntVectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) String() string {
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
func (v DenseIntVector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    buffer.WriteString(v.ConstAt(i).String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (v DenseIntVector) Export(filename string) error {
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
func (v *DenseIntVector) Import(filename string) error {
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
  *v = DenseIntVector{}
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
      *v = append(*v, int(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) MarshalJSON() ([]byte, error) {
  r := []int{}
  r = v
  return json.MarshalIndent(r, "", "  ")
}
func (v *DenseIntVector) UnmarshalJSON(data []byte) error {
  r := []int{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *v = r
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (v DenseIntVector) ConstIterator() VectorConstIterator {
  return v.ITERATOR()
}
func (v DenseIntVector) ConstIteratorFrom(i int) VectorConstIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseIntVector) Iterator() VectorIterator {
  return v.ITERATOR()
}
func (v DenseIntVector) IteratorFrom(i int) VectorIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseIntVector) JointIterator(b ConstVector) VectorJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseIntVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseIntVector) ITERATOR() *DenseIntVectorIterator {
  r := DenseIntVectorIterator{v, -1}
  r.Next()
  return &r
}
func (v DenseIntVector) ITERATOR_FROM(i int) *DenseIntVectorIterator {
  r := DenseIntVectorIterator{v, i-1}
  r.Next()
  return &r
}
func (v DenseIntVector) JOINT_ITERATOR(b ConstVector) *DenseIntVectorJointIterator {
  r := DenseIntVectorJointIterator{}
  r.it1 = v.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* const iterator
 * -------------------------------------------------------------------------- */
type DenseIntVectorIterator struct {
  v DenseIntVector
  i int
}
func (obj *DenseIntVectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseIntVectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseIntVectorIterator) GET() Int {
  return Int{&obj.v[obj.i]}
}
func (obj *DenseIntVectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseIntVectorIterator) Next() {
  obj.i++
}
func (obj *DenseIntVectorIterator) Index() int {
  return obj.i
}
func (obj *DenseIntVectorIterator) Clone() *DenseIntVectorIterator {
  return &DenseIntVectorIterator{obj.v, obj.i}
}
func (obj *DenseIntVectorIterator) CloneIterator() VectorIterator {
  return &DenseIntVectorIterator{obj.v, obj.i}
}
func (obj *DenseIntVectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseIntVectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseIntVectorJointIterator struct {
  it1 *DenseIntVectorIterator
  it2 VectorConstIterator
  idx int
  s1 Int
  s2 ConstScalar
}
func (obj *DenseIntVectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseIntVectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}
func (obj *DenseIntVectorJointIterator) Next() {
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
    obj.s2 = ConstInt(0.0)
  }
}
func (obj *DenseIntVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseIntVectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseIntVectorJointIterator) GET() (Int, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseIntVectorJointIterator) Clone() *DenseIntVectorJointIterator {
  r := DenseIntVectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseIntVectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
func (obj *DenseIntVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
