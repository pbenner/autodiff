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
import "os"
import "sort"
import "strconv"
import "strings"
/* -------------------------------------------------------------------------- */
type DenseInt8Vector []int8
/* constructors
 * -------------------------------------------------------------------------- */
func NewDenseInt8Vector(values []int8) DenseInt8Vector {
  return DenseInt8Vector(values)
}
func NullDenseInt8Vector(n int) DenseInt8Vector {
  return DenseInt8Vector(make([]int8, n))
}
// Convert vector type.
func AsDenseInt8Vector(v ConstVector) DenseInt8Vector {
  switch v_ := v.(type) {
  case DenseInt8Vector:
    return v_.Clone()
  }
  values := make([]int8, v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    values[it.Index()] = it.GetConst().GetInt8()
  }
  return NewDenseInt8Vector(values)
}
/* cloning
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) Clone() DenseInt8Vector {
  r := make([]int8, v.Dim())
  copy(r, v)
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) AT(i int) Int8 {
  return Int8{&v[i]}
}
func (v DenseInt8Vector) APPEND(w DenseInt8Vector) DenseInt8Vector {
  return append(v, w...)
}
func (v DenseInt8Vector) ToDenseInt8Matrix(n, m int) *DenseInt8Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseInt8Matrix{}
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
func (v DenseInt8Vector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseInt8Vector) At(i int) Scalar {
  return v.AT(i)
}
func (v DenseInt8Vector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i] = w.ConstAt(i).GetInt8()
  }
}
func (v DenseInt8Vector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i] = 0.0
  }
}
func (v DenseInt8Vector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseInt8Vector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseInt8Vector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseInt8Vector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    v = append(v, scalar.GetInt8())
  }
  return v
}
func (v DenseInt8Vector) AppendVector(w Vector) Vector {
  for i := 0; i < w.Dim(); i++ {
    v = append(v, w.ConstAt(i).GetInt8())
  }
  return v
}
func (v DenseInt8Vector) AsMatrix(n, m int) Matrix {
  return v.ToDenseInt8Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseInt8Vector) Dim() int {
  return len(v)
}
func (v DenseInt8Vector) Int8At(i int) int8 {
  return int8(v[i])
}
func (v DenseInt8Vector) Int16At(i int) int16 {
  return int16(v[i])
}
func (v DenseInt8Vector) Int32At(i int) int32 {
  return int32(v[i])
}
func (v DenseInt8Vector) Int64At(i int) int64 {
  return int64(v[i])
}
func (v DenseInt8Vector) IntAt(i int) int {
  return int(v[i])
}
func (v DenseInt8Vector) Float32At(i int) float32 {
  return float32(v[i])
}
func (v DenseInt8Vector) Float64At(i int) float64 {
  return float64(v[i])
}
func (v DenseInt8Vector) ConstAt(i int) ConstScalar {
  return Int8{&v[i]}
}
func (v DenseInt8Vector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f(v.At(i))
  }
}
func (v DenseInt8Vector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i] = f(v.ConstAt(i)).GetInt8()
  }
}
func (v DenseInt8Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseInt8Vector) ElementType() ScalarType {
  return Int8Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) Permute(pi []int) error {
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
type sortDenseInt8VectorByValue DenseInt8Vector
func (v sortDenseInt8VectorByValue) Len() int { return len(v) }
func (v sortDenseInt8VectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseInt8VectorByValue) Less(i, j int) bool { return v[i] < v[j] }
func (v DenseInt8Vector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseInt8VectorByValue(v)))
  } else {
    sort.Sort(sortDenseInt8VectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) String() string {
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
func (v DenseInt8Vector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v.ConstAt(i).String())
  }
  return buffer.String()
}
func (v DenseInt8Vector) Export(filename string) error {
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
func (v *DenseInt8Vector) Import(filename string) error {
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
  *v = DenseInt8Vector{}
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
      *v = append(*v, int8(value))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) MarshalJSON() ([]byte, error) {
  r := []int8{}
  r = v
  return json.MarshalIndent(r, "", "  ")
}
func (v *DenseInt8Vector) UnmarshalJSON(data []byte) error {
  r := []int8{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *v = r
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (v DenseInt8Vector) ConstIterator() VectorConstIterator {
  return v.ITERATOR()
}
func (v DenseInt8Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseInt8Vector) Iterator() VectorIterator {
  return v.ITERATOR()
}
func (v DenseInt8Vector) IteratorFrom(i int) VectorIterator {
  return v.ITERATOR_FROM(i)
}
func (v DenseInt8Vector) JointIterator(b ConstVector) VectorJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseInt8Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return v.JOINT_ITERATOR(b)
}
func (v DenseInt8Vector) ITERATOR() *DenseInt8VectorIterator {
  r := DenseInt8VectorIterator{v, -1}
  r.Next()
  return &r
}
func (v DenseInt8Vector) ITERATOR_FROM(i int) *DenseInt8VectorIterator {
  r := DenseInt8VectorIterator{v, i-1}
  r.Next()
  return &r
}
func (v DenseInt8Vector) JOINT_ITERATOR(b ConstVector) *DenseInt8VectorJointIterator {
  r := DenseInt8VectorJointIterator{}
  r.it1 = v.ITERATOR()
  r.it2 = b.ConstIterator()
  r.idx = -1
  r.Next()
  return &r
}
/* const iterator
 * -------------------------------------------------------------------------- */
type DenseInt8VectorIterator struct {
  v DenseInt8Vector
  i int
}
func (obj *DenseInt8VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseInt8VectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseInt8VectorIterator) GET() Int8 {
  return Int8{&obj.v[obj.i]}
}
func (obj *DenseInt8VectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseInt8VectorIterator) Next() {
  obj.i++
}
func (obj *DenseInt8VectorIterator) Index() int {
  return obj.i
}
func (obj *DenseInt8VectorIterator) Clone() *DenseInt8VectorIterator {
  return &DenseInt8VectorIterator{obj.v, obj.i}
}
func (obj *DenseInt8VectorIterator) CloneIterator() VectorIterator {
  return &DenseInt8VectorIterator{obj.v, obj.i}
}
func (obj *DenseInt8VectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseInt8VectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseInt8VectorJointIterator struct {
  it1 *DenseInt8VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Int8
  s2 ConstScalar
}
func (obj *DenseInt8VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseInt8VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}
func (obj *DenseInt8VectorJointIterator) Next() {
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
    obj.s2 = ConstInt8(0.0)
  }
}
func (obj *DenseInt8VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt8VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseInt8VectorJointIterator) GET() (Int8, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *DenseInt8VectorJointIterator) Clone() *DenseInt8VectorJointIterator {
  r := DenseInt8VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseInt8VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
func (obj *DenseInt8VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
