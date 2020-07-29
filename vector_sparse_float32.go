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
type SparseFloat32Vector struct {
  vectorSparseIndex
  values map[int]Float32
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseFloat32Vector(indices []int, values []float32, n int) *SparseFloat32Vector {
  if len(indices) != len(values) {
    panic("number of indices does not match number of values")
  }
  r := nilSparseFloat32Vector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if _, ok := r.values[k]; ok {
      panic("index appeared multiple times")
    } else {
      if values[i] != 0.0 {
        r.values[k] = NewFloat32(values[i])
        r.indexInsert(k)
      }
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseFloat32Vector(length int) *SparseFloat32Vector {
  v := nilSparseFloat32Vector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseFloat32Vector(length int) *SparseFloat32Vector {
  return &SparseFloat32Vector{values: make(map[int]Float32), n: length}
}
// Convert vector type.
func AsSparseFloat32Vector(v ConstVector) *SparseFloat32Vector {
  switch v_ := v.(type) {
  case *SparseFloat32Vector:
    return v_.Clone()
  }
  r := NullSparseFloat32Vector(v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    r.AT(it.Index()).Set(it.GetConst())
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseFloat32Vector) Clone() *SparseFloat32Vector {
  r := nilSparseFloat32Vector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndex = obj.indexClone()
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) AT(i int) Float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullFloat32()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseFloat32Vector) AT_(i int) Float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return Float32{}
  }
}
func (obj *SparseFloat32Vector) SET(x *SparseFloat32Vector) {
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
    case s1.ptr != nil : s1.SetFloat32(0)
    default : obj.AT(it.Index()).SET(s2)
    }
  }
}
func (obj *SparseFloat32Vector) SLICE(i, j int) *SparseFloat32Vector {
  r := nilSparseFloat32Vector(j-i)
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
func (obj *SparseFloat32Vector) APPEND(w *SparseFloat32Vector) *SparseFloat32Vector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for it := w.ITERATOR(); it.Ok(); it.Next() {
    i := obj.n+it.Index()
    r.values[i] = it.GET()
    r.indexInsert(i)
  }
  return r
}
func (obj *SparseFloat32Vector) ToSparseFloat32Matrix(n, m int) *SparseFloat32Matrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullSparseFloat32Vector(obj.n)
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    v.At(it.Index()).Set(it.GET())
  }
  matrix := SparseFloat32Matrix{}
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
func (obj *SparseFloat32Vector) CloneVector() Vector {
  return obj.Clone()
}
func (obj *SparseFloat32Vector) At(i int) Scalar {
  return obj.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseFloat32Vector) Set(x ConstVector) {
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
    case s1.ptr != nil : s1.SetFloat32(0)
    default : obj.AT(it.Index()).Set(s2)
    }
  }
}
func (obj *SparseFloat32Vector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseFloat32Vector) ReverseOrder() {
  n := obj.Dim()
  values := make(map[int]Float32)
  index := vectorSparseIndex{}
  for i, s := range obj.values {
    j := n-i-1
    values[j] = s
    index.indexInsert(j)
  }
  obj.values = values
  obj.vectorSparseIndex = index
}
func (obj *SparseFloat32Vector) Slice(i, j int) Vector {
  return obj.SLICE(i, j)
}
func (obj *SparseFloat32Vector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
}
func (obj *SparseFloat32Vector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case Float32:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertScalar(Float32Type).(Float32)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseFloat32Vector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseFloat32Vector:
    return obj.APPEND(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for it := w.Iterator(); it.Ok(); it.Next() {
      r.values[obj.n+it.Index()] = it.Get().ConvertScalar(Float32Type).(Float32)
      r.indexInsert(obj.n+it.Index())
    }
    return r
  }
}
func (v *SparseFloat32Vector) AsMatrix(n, m int) Matrix {
  return v.ToSparseFloat32Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj *SparseFloat32Vector) Dim() int {
  return obj.n
}
func (obj *SparseFloat32Vector) Int8At(i int) int8 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt8()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) Int16At(i int) int16 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt16()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) Int32At(i int) int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt32()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) Int64At(i int) int64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt64()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) IntAt(i int) int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) Float32At(i int) float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat32()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) Float64At(i int) float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat64()
  } else {
    return 0
  }
}
func (obj *SparseFloat32Vector) ConstAt(i int) ConstScalar {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstFloat32(0.0)
  }
}
func (obj *SparseFloat32Vector) ConstSlice(i, j int) ConstVector {
  return obj.SLICE(i, j)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseFloat32Vector) MapSet(f func(ConstScalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseFloat32Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseFloat32Vector) ElementType() ScalarType {
  return Float32Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) Permute(pi []int) error {
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
type sortSparseFloat32VectorByValue struct {
  Value []Float32
}
func (obj sortSparseFloat32VectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseFloat32VectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseFloat32VectorByValue) Less(i, j int) bool {
  return obj.Value[i].Smaller(obj.Value[j])
}
func (obj *SparseFloat32Vector) Sort(reverse bool) {
  r := sortSparseFloat32VectorByValue{}
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
  obj.values = make(map[int]Float32)
  obj.vectorSparseIndex = vectorSparseIndex{}
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseFloat32VectorByValue(r))
  }
  for i := 0; i < len(r.Value); i++ {
    if r.Value[i].GetFloat32() > 0.0 {
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
func (obj *SparseFloat32Vector) String() string {
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
func (obj *SparseFloat32Vector) Table() string {
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
    buffer.WriteString(fmt.Sprintf("%d:%s", i, ConstFloat32(0.0)))
  }
  return buffer.String()
}
func (obj *SparseFloat32Vector) Export(filename string) error {
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
func (obj *SparseFloat32Vector) Import(filename string) error {
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
  values := []float32{}
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
        values = append(values, float32(v))
      }
    }
  }
  *obj = *NewSparseFloat32Vector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []float32{}
  r := struct{
    Index []int
    Value []float32
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetFloat32())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseFloat32Vector) UnmarshalJSON(data []byte) error {
  r := struct{
    Index []int
    Value []float32
    Length int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  if len(r.Index) != len(r.Value) {
    return fmt.Errorf("invalid sparse vector")
  }
  *obj = *NewSparseFloat32Vector(r.Index, r.Value, r.Length)
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat32Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj *SparseFloat32Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseFloat32Vector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj *SparseFloat32Vector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseFloat32Vector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseFloat32Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseFloat32Vector) ITERATOR() *SparseFloat32VectorIterator {
  r := SparseFloat32VectorIterator{obj.indexIterator(), obj}
  r.skip()
  return &r
}
func (obj *SparseFloat32Vector) ITERATOR_FROM(i int) *SparseFloat32VectorIterator {
  r := SparseFloat32VectorIterator{obj.indexIteratorFrom(i), obj}
  r.skip()
  return &r
}
func (obj *SparseFloat32Vector) JOINT_ITERATOR(b ConstVector) *SparseFloat32VectorJointIterator {
  r := SparseFloat32VectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, Float32{}, nil}
  r.Next()
  return &r
}
func (obj *SparseFloat32Vector) JOINT3_ITERATOR(b, c ConstVector) *SparseFloat32VectorJoint3Iterator {
  r := SparseFloat32VectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, Float32{}, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseFloat32Vector) JOINT_ITERATOR_(b *SparseFloat32Vector) *SparseFloat32VectorJointIterator_ {
  r := SparseFloat32VectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, Float32{}, Float32{}}
  r.Next()
  return &r
}
func (obj *SparseFloat32Vector) JOINT3_ITERATOR_(b, c *SparseFloat32Vector) *SparseFloat32VectorJoint3Iterator_ {
  r := SparseFloat32VectorJoint3Iterator_{obj.ITERATOR(), b.ITERATOR(), c.ITERATOR(), -1, Float32{}, Float32{}, Float32{}}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseFloat32VectorIterator struct {
  vectorSparseIndexIterator
  v *SparseFloat32Vector
}
func (obj *SparseFloat32VectorIterator) Get() Scalar {
  if v := obj.GET(); v.ptr == nil {
    return nil
  } else {
    return v
  }
}
func (obj *SparseFloat32VectorIterator) GetConst() ConstScalar {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return nil
  }
}
func (obj *SparseFloat32VectorIterator) GET() Float32 {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return Float32{}
  }
}
func (obj *SparseFloat32VectorIterator) Next() {
  obj.vectorSparseIndexIterator.Next()
  obj.skip()
}
func (obj *SparseFloat32VectorIterator) skip() {
  for obj.Ok() && obj.GET().nullScalar() {
    i := obj.Index()
    obj.vectorSparseIndexIterator.Next()
    delete(obj.v.values, i)
    obj.v.indexDelete(i)
  }
}
func (obj *SparseFloat32VectorIterator) Index() int {
  return obj.vectorSparseIndexIterator.Get()
}
func (obj *SparseFloat32VectorIterator) Clone() *SparseFloat32VectorIterator {
  return &SparseFloat32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseFloat32VectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseFloat32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseFloat32VectorIterator) CloneIterator() VectorIterator {
  return &SparseFloat32VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat32VectorJointIterator struct {
  it1 *SparseFloat32VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Float32
  s2 ConstScalar
}
func (obj *SparseFloat32VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseFloat32VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat32() == float32(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat32() == float32(0))
}
func (obj *SparseFloat32VectorJointIterator) Next() {
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
    obj.s2 = ConstFloat32(0.0)
  }
}
func (obj *SparseFloat32VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseFloat32VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseFloat32VectorJointIterator) GET() (Float32, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseFloat32VectorJointIterator) Clone() *SparseFloat32VectorJointIterator {
  r := SparseFloat32VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *SparseFloat32VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *SparseFloat32VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat32VectorJoint3Iterator struct {
  it1 *SparseFloat32VectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 Float32
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseFloat32VectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseFloat32VectorJoint3Iterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat32() == float32(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat32() == float32(0)) ||
         !(obj.s3 == nil || obj.s3.GetFloat32() == float32(0))
}
func (obj *SparseFloat32VectorJoint3Iterator) Next() {
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
    obj.s2 = ConstFloat32(0.0)
  }
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstFloat32(0.0)
  }
}
func (obj *SparseFloat32VectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2, obj.s3
  } else {
    return obj.s1, obj.s2, obj.s3
  }
}
func (obj *SparseFloat32VectorJoint3Iterator) GET() (Float32, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat32VectorJointIterator_ struct {
  it1 *SparseFloat32VectorIterator
  it2 *SparseFloat32VectorIterator
  idx int
  s1 Float32
  s2 Float32
}
func (obj *SparseFloat32VectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *SparseFloat32VectorJointIterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil
}
func (obj *SparseFloat32VectorJointIterator_) Next() {
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
func (obj *SparseFloat32VectorJointIterator_) GET() (Float32, Float32) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat32VectorJoint3Iterator_ struct {
  it1 *SparseFloat32VectorIterator
  it2 *SparseFloat32VectorIterator
  it3 *SparseFloat32VectorIterator
  idx int
  s1 Float32
  s2 Float32
  s3 Float32
}
func (obj *SparseFloat32VectorJoint3Iterator_) Index() int {
  return obj.idx
}
func (obj *SparseFloat32VectorJoint3Iterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil || obj.s3.ptr != nil
}
func (obj *SparseFloat32VectorJoint3Iterator_) Next() {
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
func (obj *SparseFloat32VectorJoint3Iterator_) GET() (Float32, Float32, Float32) {
  return obj.s1, obj.s2, obj.s3
}
