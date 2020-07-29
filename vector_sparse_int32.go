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
import "errors"
import "io"
import "os"
import "sort"
import "strconv"
import "strings"
/* vector type declaration
 * -------------------------------------------------------------------------- */
type SparseInt32Vector struct {
  vectorSparseIndex
  values map[int]Int32
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseInt32Vector(indices []int, values []int32, n int) *SparseInt32Vector {
  if len(indices) != len(values) {
    panic("number of indices does not match number of values")
  }
  r := nilSparseInt32Vector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if _, ok := r.values[k]; ok {
      panic("index appeared multiple times")
    } else {
      if values[i] != 0.0 {
        r.values[k] = NewInt32(values[i])
        r.indexInsert(k)
      }
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseInt32Vector(length int) *SparseInt32Vector {
  v := nilSparseInt32Vector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseInt32Vector(length int) *SparseInt32Vector {
  return &SparseInt32Vector{values: make(map[int]Int32), n: length}
}
// Convert vector type.
func AsSparseInt32Vector(v ConstVector) *SparseInt32Vector {
  switch v_ := v.(type) {
  case *SparseInt32Vector:
    return v_.Clone()
  }
  r := NullSparseInt32Vector(v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    r.AT(it.Index()).Set(it.GetConst())
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseInt32Vector) Clone() *SparseInt32Vector {
  r := nilSparseInt32Vector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndex = obj.indexClone()
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) AT(i int) Int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullInt32()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseInt32Vector) AT_(i int) Int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return Int32{}
  }
}
func (obj *SparseInt32Vector) SET(x *SparseInt32Vector) {
  if obj == x {
    return
  }
  if obj.Dim() != x.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for it := obj.JOINT_ITERATOR_(x); it.Ok(); it.Next() {
    s1, s2 := it.GET()
    switch {
    case s1.ptr != nil && s2.ptr != nil: s1.SET(s2)
    case s1.ptr != nil : s1.SetInt32(0)
    default : obj.AT(it.Index()).SET(s2)
    }
  }
}
func (obj *SparseInt32Vector) SLICE(i, j int) *SparseInt32Vector {
  r := nilSparseInt32Vector(j-i)
  for it := obj.indexIteratorFrom(i); it.Ok(); it.Next() {
    if it.Get() >= j {
      break
    }
    k := it.Get()
    r.values[k-i] = obj.values[k]
    r.indexInsert(k-i)
  }
  return r
}
func (obj *SparseInt32Vector) APPEND(w *SparseInt32Vector) *SparseInt32Vector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for it := w.ITERATOR(); it.Ok(); it.Next() {
    i := obj.n+it.Index()
    r.values[i] = it.GET()
    r.indexInsert(i)
  }
  return r
}
func (obj *SparseInt32Vector) ToSparseInt32Matrix(n, m int) *SparseInt32Matrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullSparseInt32Vector(obj.n)
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    v.At(it.Index()).Set(it.GET())
  }
  matrix := SparseInt32Matrix{}
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
/* vector interface
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) CloneVector() Vector {
  return obj.Clone()
}
func (obj *SparseInt32Vector) At(i int) Scalar {
  return obj.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseInt32Vector) Set(x ConstVector) {
  if obj == x {
    return
  }
  if obj.Dim() != x.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for it := obj.JOINT_ITERATOR(x); it.Ok(); it.Next() {
    s1, s2 := it.GET()
    switch {
    case s1.ptr != nil && s2 != nil: s1.Set(s2)
    case s1.ptr != nil : s1.SetInt32(0)
    default : obj.AT(it.Index()).Set(s2)
    }
  }
}
func (obj *SparseInt32Vector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseInt32Vector) ReverseOrder() {
  n := obj.Dim()
  values := make(map[int]Int32)
  index := vectorSparseIndex{}
  for i, s := range obj.values {
    j := n-i-1
    values[j] = s
    index.indexInsert(j)
  }
  obj.values = values
  obj.vectorSparseIndex = index
}
func (obj *SparseInt32Vector) Slice(i, j int) Vector {
  return obj.SLICE(i, j)
}
func (obj *SparseInt32Vector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
}
func (obj *SparseInt32Vector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case Int32:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertScalar(Int32Type).(Int32)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseInt32Vector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseInt32Vector:
    return obj.APPEND(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for it := w.Iterator(); it.Ok(); it.Next() {
      r.values[obj.n+it.Index()] = it.Get().ConvertScalar(Int32Type).(Int32)
      r.indexInsert(obj.n+it.Index())
    }
    return r
  }
}
func (v *SparseInt32Vector) AsMatrix(n, m int) Matrix {
  return v.ToSparseInt32Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj *SparseInt32Vector) Dim() int {
  return obj.n
}
func (obj *SparseInt32Vector) Int8At(i int) int8 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt8()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) Int16At(i int) int16 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt16()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) Int32At(i int) int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt32()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) Int64At(i int) int64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt64()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) IntAt(i int) int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) Float32At(i int) float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat32()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) Float64At(i int) float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat64()
  } else {
    return 0
  }
}
func (obj *SparseInt32Vector) ConstAt(i int) ConstScalar {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstInt32(0.0)
  }
}
func (obj *SparseInt32Vector) ConstSlice(i, j int) ConstVector {
  return obj.SLICE(i, j)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseInt32Vector) MapSet(f func(ConstScalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseInt32Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseInt32Vector) ElementType() ScalarType {
  return Int32Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) Permute(pi []int) error {
  if len(pi) != obj.n {
    return errors.New("Permute(): permutation vector has invalid length!")
  }
  // permute vector
  for i := 0; i < obj.n; i++ {
    if pi[i] < 0 || pi[i] >= obj.n {
      return errors.New("Permute(): invalid permutation")
    }
    if i != pi[i] && pi[i] > i {
      // permute elements
      _, ok1 := obj.values[i]
      _, ok2 := obj.values[pi[i]]
      if ok1 && ok2 {
        obj.values[pi[i]], obj.values[i] = obj.values[i], obj.values[pi[i]]
      } else
      if ok1 {
        obj.values[pi[i]] = obj.values[i]
        delete(obj.values, i)
      } else
      if ok2 {
        obj.values[i] = obj.values[pi[i]]
        delete(obj.values, pi[i])
      }
    }
  }
  obj.vectorSparseIndex = vectorSparseIndex{}
  for i := 0; i < len(pi); i++ {
    obj.indexInsert(pi[i])
  }
  return nil
}
/* sorting
 * -------------------------------------------------------------------------- */
type sortSparseInt32VectorByValue struct {
  Value []Int32
}
func (obj sortSparseInt32VectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseInt32VectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseInt32VectorByValue) Less(i, j int) bool {
  return obj.Value[i].Smaller(obj.Value[j])
}
func (obj *SparseInt32Vector) Sort(reverse bool) {
  r := sortSparseInt32VectorByValue{}
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    r.Value = append(r.Value, it.GET())
  }
  ip := 0
  in := 0
  if reverse {
    in = obj.n - len(obj.values)
  } else {
    ip = obj.n - len(obj.values)
  }
  obj.values = make(map[int]Int32)
  obj.vectorSparseIndex = vectorSparseIndex{}
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseInt32VectorByValue(r))
  }
  for i := 0; i < len(r.Value); i++ {
    if r.Value[i].GetInt32() > 0.0 {
      // copy negative values
      obj.values[i+ip] = r.Value[i]
      obj.indexInsert(i+ip)
    } else {
      // copy negative values
      obj.values[i+in] = r.Value[i]
      obj.indexInsert(i+in)
    }
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) String() string {
  var buffer bytes.Buffer
  buffer.WriteString(fmt.Sprintf("%d:[", obj.n))
  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(", ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", it.Index(), it.GetConst()))
  }
  buffer.WriteString("]")
  return buffer.String()
}
func (obj *SparseInt32Vector) Table() string {
  var buffer bytes.Buffer
  first := true
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    if !first {
      buffer.WriteString(" ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", it.Index(), it.GetConst()))
  }
  if _, ok := obj.values[obj.n-1]; !ok {
    i := obj.n-1
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", i, ConstInt32(0.0)))
  }
  return buffer.String()
}
func (obj *SparseInt32Vector) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()
  w := bufio.NewWriter(f)
  defer w.Flush()
  if _, err := fmt.Fprintf(w, "%s\n", obj.Table()); err != nil {
    return err
  }
  return nil
}
func (obj *SparseInt32Vector) Import(filename string) error {
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
  values := []int32{}
  indices := []int{}
  n := 0
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
      split := strings.Split(fields[i], ":")
      if len(split) != 2 {
        return fmt.Errorf("invalid sparse table")
      }
      // parse index
      if k, err := strconv.ParseInt(split[0], 10, 64); err != nil {
        return fmt.Errorf("invalid sparse table")
      } else {
        indices = append(indices, int(k))
        // update vector length length
        if int(k)+1 > n {
          n = int(k)+1
        }
      }
      // parse value
      if v, err := strconv.ParseFloat(split[1], 64); err != nil {
        return fmt.Errorf("invalid sparse table")
      } else {
        values = append(values, int32(v))
      }
    }
  }
  *obj = *NewSparseInt32Vector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []int32{}
  r := struct{
    Index []int
    Value []int32
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetInt32())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseInt32Vector) UnmarshalJSON(data []byte) error {
  r := struct{
    Index []int
    Value []int32
    Length int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  if len(r.Index) != len(r.Value) {
    return fmt.Errorf("invalid sparse vector")
  }
  *obj = *NewSparseInt32Vector(r.Index, r.Value, r.Length)
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseInt32Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj *SparseInt32Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseInt32Vector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj *SparseInt32Vector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseInt32Vector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseInt32Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseInt32Vector) ITERATOR() *SparseInt32VectorIterator {
  r := SparseInt32VectorIterator{obj.indexIterator(), obj}
  r.skip()
  return &r
}
func (obj *SparseInt32Vector) ITERATOR_FROM(i int) *SparseInt32VectorIterator {
  r := SparseInt32VectorIterator{obj.indexIteratorFrom(i), obj}
  r.skip()
  return &r
}
func (obj *SparseInt32Vector) JOINT_ITERATOR(b ConstVector) *SparseInt32VectorJointIterator {
  r := SparseInt32VectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, Int32{}, nil}
  r.Next()
  return &r
}
func (obj *SparseInt32Vector) JOINT3_ITERATOR(b, c ConstVector) *SparseInt32VectorJoint3Iterator {
  r := SparseInt32VectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, Int32{}, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseInt32Vector) JOINT_ITERATOR_(b *SparseInt32Vector) *SparseInt32VectorJointIterator_ {
  r := SparseInt32VectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, Int32{}, Int32{}}
  r.Next()
  return &r
}
func (obj *SparseInt32Vector) JOINT3_ITERATOR_(b, c *SparseInt32Vector) *SparseInt32VectorJoint3Iterator_ {
  r := SparseInt32VectorJoint3Iterator_{obj.ITERATOR(), b.ITERATOR(), c.ITERATOR(), -1, Int32{}, Int32{}, Int32{}}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseInt32VectorIterator struct {
  vectorSparseIndexIterator
  v *SparseInt32Vector
}
func (obj *SparseInt32VectorIterator) Get() Scalar {
  if v := obj.GET(); v.ptr == nil {
    return nil
  } else {
    return v
  }
}
func (obj *SparseInt32VectorIterator) GetConst() ConstScalar {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return nil
  }
}
func (obj *SparseInt32VectorIterator) GET() Int32 {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return Int32{}
  }
}
func (obj *SparseInt32VectorIterator) Next() {
  obj.vectorSparseIndexIterator.Next()
  obj.skip()
}
func (obj *SparseInt32VectorIterator) skip() {
  for obj.Ok() && obj.GET().nullScalar() {
    i := obj.Index()
    obj.vectorSparseIndexIterator.Next()
    delete(obj.v.values, i)
    obj.v.indexDelete(i)
  }
}
func (obj *SparseInt32VectorIterator) Index() int {
  return obj.vectorSparseIndexIterator.Get()
}
func (obj *SparseInt32VectorIterator) Clone() *SparseInt32VectorIterator {
  return &SparseInt32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseInt32VectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseInt32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseInt32VectorIterator) CloneIterator() VectorIterator {
  return &SparseInt32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseInt32VectorJointIterator struct {
  it1 *SparseInt32VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Int32
  s2 ConstScalar
}
func (obj *SparseInt32VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseInt32VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt32() == int32(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt32() == int32(0))
}
func (obj *SparseInt32VectorJointIterator) Next() {
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
func (obj *SparseInt32VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseInt32VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseInt32VectorJointIterator) GET() (Int32, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseInt32VectorJointIterator) Clone() *SparseInt32VectorJointIterator {
  r := SparseInt32VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *SparseInt32VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *SparseInt32VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseInt32VectorJoint3Iterator struct {
  it1 *SparseInt32VectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 Int32
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseInt32VectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseInt32VectorJoint3Iterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt32() == int32(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt32() == int32(0)) ||
         !(obj.s3 == nil || obj.s3.GetInt32() == int32(0))
}
func (obj *SparseInt32VectorJoint3Iterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  ok3 := obj.it3.Ok()
  obj.s1.ptr = nil
  obj.s2 = nil
  obj.s3 = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    i := obj.it2.Index()
    switch {
    case obj.idx > i || !ok1:
      obj.idx = i
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GetConst()
    case obj.idx == i:
      obj.s2 = obj.it2.GetConst()
    }
  }
  if ok3 {
    i := obj.it3.Index()
    switch {
    case obj.idx > i || (!ok1 && !ok2):
      obj.idx = i
      obj.s1.ptr = nil
      obj.s2 = nil
      obj.s3 = obj.it3.GetConst()
    case obj.idx == i:
      obj.s3 = obj.it3.GetConst()
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
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstInt32(0.0)
  }
}
func (obj *SparseInt32VectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2, obj.s3
  } else {
    return obj.s1, obj.s2, obj.s3
  }
}
func (obj *SparseInt32VectorJoint3Iterator) GET() (Int32, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseInt32VectorJointIterator_ struct {
  it1 *SparseInt32VectorIterator
  it2 *SparseInt32VectorIterator
  idx int
  s1 Int32
  s2 Int32
}
func (obj *SparseInt32VectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *SparseInt32VectorJointIterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil
}
func (obj *SparseInt32VectorJointIterator_) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1.ptr = nil
  obj.s2.ptr = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GET()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GET()
    }
  }
  if obj.s1.ptr != nil {
    obj.it1.Next()
  }
  if obj.s2.ptr != nil {
    obj.it2.Next()
  }
}
func (obj *SparseInt32VectorJointIterator_) GET() (Int32, Int32) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseInt32VectorJoint3Iterator_ struct {
  it1 *SparseInt32VectorIterator
  it2 *SparseInt32VectorIterator
  it3 *SparseInt32VectorIterator
  idx int
  s1 Int32
  s2 Int32
  s3 Int32
}
func (obj *SparseInt32VectorJoint3Iterator_) Index() int {
  return obj.idx
}
func (obj *SparseInt32VectorJoint3Iterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil || obj.s3.ptr != nil
}
func (obj *SparseInt32VectorJoint3Iterator_) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  ok3 := obj.it3.Ok()
  obj.s1.ptr = nil
  obj.s2.ptr = nil
  obj.s3.ptr = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    i := obj.it2.Index()
    switch {
    case obj.idx > i || !ok1:
      obj.idx = i
      obj.s1.ptr = nil
      obj.s2 = obj.it2.GET()
    case obj.idx == i:
      obj.s2 = obj.it2.GET()
    }
  }
  if ok3 {
    i := obj.it3.Index()
    switch {
    case obj.idx > i || (!ok1 && !ok2):
      obj.idx = i
      obj.s1.ptr = nil
      obj.s2.ptr = nil
      obj.s3 = obj.it3.GET()
    case obj.idx == i:
      obj.s3 = obj.it3.GET()
    }
  }
  if obj.s1.ptr != nil {
    obj.it1.Next()
  }
  if obj.s2.ptr != nil {
    obj.it2.Next()
  }
  if obj.s3.ptr != nil {
    obj.it3.Next()
  }
}
func (obj *SparseInt32VectorJoint3Iterator_) GET() (Int32, Int32, Int32) {
  return obj.s1, obj.s2, obj.s3
}
