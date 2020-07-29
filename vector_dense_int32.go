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
type DenseInt32Vector []int32
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseInt32Vector(values []int32) DenseInt32Vector {
  return DenseInt32Vector(values)
}
func NullDenseInt32Vector(n int) DenseInt32Vector {
  return DenseInt32Vector(make([]int32, n))
}
// Convert vector type.
func AsDenseInt32Vector(v ConstVector) DenseInt32Vector {
  switch v_ := v.(type) {
  case DenseInt32Vector:
    return v_.Clone()
  }
  values := make([]int32, v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    values[it.Index()] = it.GetConst().GetInt32()
  }
  return NewDenseInt32Vector(values)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) Clone() DenseInt32Vector {
  r := make([]int32, v.Dim())
  copy(r, v)
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) AT(i int) Int32 {
  return Int32{&v[i]}
}
func (v DenseInt32Vector) APPEND(w DenseInt32Vector) DenseInt32Vector {
  return append(v, w...)
}
func (v DenseInt32Vector) ToDenseInt32Matrix(n, m int) *DenseInt32Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseInt32Matrix{}
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
func (v DenseInt32Vector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseInt32Vector) At(i int) Scalar {
  return v.AT(i)
}
func (v DenseInt32Vector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i] = w.ConstAt(i).GetInt32()
  }
}
func (v DenseInt32Vector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i] = 0.0
  }
}
func (v DenseInt32Vector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseInt32Vector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseInt32Vector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseInt32Vector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    v = append(v, scalar.GetInt32())
  }
  return v
}
func (v DenseInt32Vector) AppendVector(w Vector) Vector {
  for i := 0; i < w.Dim(); i++ {
    v = append(v, w.ConstAt(i).GetInt32())
  }
  return v
}
func (v DenseInt32Vector) AsMatrix(n, m int) Matrix {
  return v.ToDenseInt32Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseInt32Vector) Dim() int {
  return len(v)
}
func (v DenseInt32Vector) Int8At(i int) int8 {
  return int8(v[i])
}
func (v DenseInt32Vector) Int16At(i int) int16 {
  return int16(v[i])
}
func (v DenseInt32Vector) Int32At(i int) int32 {
  return int32(v[i])
}
func (v DenseInt32Vector) Int64At(i int) int64 {
  return int64(v[i])
}
func (v DenseInt32Vector) IntAt(i int) int {
  return int(v[i])
}
func (v DenseInt32Vector) Float32At(i int) float32 {
  return float32(v[i])
}
func (v DenseInt32Vector) Float64At(i int) float64 {
  return float64(v[i])
}
func (v DenseInt32Vector) ConstAt(i int) ConstScalar {
  return Int32{&v[i]}
}
func (v DenseInt32Vector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(v.At(i))
  }
}
func (v DenseInt32Vector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i] = f(v.ConstAt(i)).GetInt32()
  }
}
func (v DenseInt32Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseInt32Vector) ElementType() ScalarType {
  return Int32Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) Permute(pi []int) error {
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
type sortDenseInt32VectorByValue DenseInt32Vector
func (v sortDenseInt32VectorByValue) Len() int { return len(v) }
func (v sortDenseInt32VectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseInt32VectorByValue) Less(i, j int) bool { return v[i] < v[j] }
func (v DenseInt32Vector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseInt32VectorByValue(v)))
  } else {
    sort.Sort(sortDenseInt32VectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) String() string {
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
func (v DenseInt32Vector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    buffer.WriteString(v.ConstAt(i).String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (v DenseInt32Vector) Export(filename string) error {
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
func (v *DenseInt32Vector) Import(filename string) error {
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
  *v = DenseInt32Vector{}
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
      *v = append(*v, int32(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) MarshalJSON() ([]byte, error) {
  r := []int32{}
  r = v
  return json.MarshalIndent(r, "", "  ")
}
func (v *DenseInt32Vector) UnmarshalJSON(data []byte) error {
  r := []int32{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *v = r
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (v DenseInt32Vector) ConstIterator() VectorConstIterator {
  return v.ITERATOR()
}
func (v DenseInt32Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseInt32Vector) Iterator() VectorIterator {
  return v.ITERATOR()
}
func (v DenseInt32Vector) IteratorFrom(i int) VectorIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseInt32Vector) JointIterator(b ConstVector) VectorJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseInt32Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseInt32Vector) ITERATOR() *DenseInt32VectorIterator {
  r := DenseInt32VectorIterator{v, -1}
  r.Next()
  return &r
}
func (v DenseInt32Vector) ITERATOR_FROM(i int) *DenseInt32VectorIterator {
  r := DenseInt32VectorIterator{v, i-1}
  r.Next()
  return &r
}
func (v DenseInt32Vector) JOINT_ITERATOR(b ConstVector) *DenseInt32VectorJointIterator {
  r := DenseInt32VectorJointIterator{}
  r.it1 = v.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* const iterator
 * -------------------------------------------------------------------------- */
type DenseInt32VectorIterator struct {
  v DenseInt32Vector
  i int
}
func (obj *DenseInt32VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseInt32VectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseInt32VectorIterator) GET() Int32 {
  return Int32{&obj.v[obj.i]}
}
func (obj *DenseInt32VectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseInt32VectorIterator) Next() {
  obj.i++
}
func (obj *DenseInt32VectorIterator) Index() int {
  return obj.i
}
func (obj *DenseInt32VectorIterator) Clone() *DenseInt32VectorIterator {
  return &DenseInt32VectorIterator{obj.v, obj.i}
}
func (obj *DenseInt32VectorIterator) CloneIterator() VectorIterator {
  return &DenseInt32VectorIterator{obj.v, obj.i}
}
func (obj *DenseInt32VectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseInt32VectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseInt32VectorJointIterator struct {
  it1 *DenseInt32VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Int32
  s2 ConstScalar
}
func (obj *DenseInt32VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseInt32VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}
func (obj *DenseInt32VectorJointIterator) Next() {
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
    obj.s2 = ConstInt32(0.0)
  }
}
func (obj *DenseInt32VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt32VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt32VectorJointIterator) GET() (Int32, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseInt32VectorJointIterator) Clone() *DenseInt32VectorJointIterator {
  r := DenseInt32VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseInt32VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
func (obj *DenseInt32VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
