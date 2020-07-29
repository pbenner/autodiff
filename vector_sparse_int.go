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
type SparseIntVector struct {
  vectorSparseIndex
  values map[int]Int
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseIntVector(indices []int, values []int, n int) *SparseIntVector {
  if len(indices) != len(values) {
    panic("number of indices does not match number of values")
  }
  r := nilSparseIntVector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if _, ok := r.values[k]; ok {
      panic("index appeared multiple times")
    } else {
      if values[i] != 0.0 {
        r.values[k] = NewInt(values[i])
        r.indexInsert(k)
      }
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseIntVector(length int) *SparseIntVector {
  v := nilSparseIntVector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseIntVector(length int) *SparseIntVector {
  return &SparseIntVector{values: make(map[int]Int), n: length}
}
// Convert vector type.
func AsSparseIntVector(v ConstVector) *SparseIntVector {
  switch v_ := v.(type) {
  case *SparseIntVector:
    return v_.Clone()
  }
  r := NullSparseIntVector(v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    r.AT(it.Index()).Set(it.GetConst())
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseIntVector) Clone() *SparseIntVector {
  r := nilSparseIntVector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndex = obj.indexClone()
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) AT(i int) Int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullInt()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseIntVector) AT_(i int) Int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return Int{}
  }
}
func (obj *SparseIntVector) SET(x *SparseIntVector) {
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
    case s1.ptr != nil : s1.SetInt(0)
    default : obj.AT(it.Index()).SET(s2)
    }
  }
}
func (obj *SparseIntVector) SLICE(i, j int) *SparseIntVector {
  r := nilSparseIntVector(j-i)
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
func (obj *SparseIntVector) APPEND(w *SparseIntVector) *SparseIntVector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for it := w.ITERATOR(); it.Ok(); it.Next() {
    i := obj.n+it.Index()
    r.values[i] = it.GET()
    r.indexInsert(i)
  }
  return r
}
func (obj *SparseIntVector) ToSparseIntMatrix(n, m int) *SparseIntMatrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullSparseIntVector(obj.n)
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    v.At(it.Index()).Set(it.GET())
  }
  matrix := SparseIntMatrix{}
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
func (obj *SparseIntVector) CloneVector() Vector {
  return obj.Clone()
}
func (obj *SparseIntVector) At(i int) Scalar {
  return obj.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseIntVector) Set(x ConstVector) {
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
    case s1.ptr != nil : s1.SetInt(0)
    default : obj.AT(it.Index()).Set(s2)
    }
  }
}
func (obj *SparseIntVector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseIntVector) ReverseOrder() {
  n := obj.Dim()
  values := make(map[int]Int)
  index := vectorSparseIndex{}
  for i, s := range obj.values {
    j := n-i-1
    values[j] = s
    index.indexInsert(j)
  }
  obj.values = values
  obj.vectorSparseIndex = index
}
func (obj *SparseIntVector) Slice(i, j int) Vector {
  return obj.SLICE(i, j)
}
func (obj *SparseIntVector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
}
func (obj *SparseIntVector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case Int:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertScalar(IntType).(Int)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseIntVector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseIntVector:
    return obj.APPEND(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for it := w.Iterator(); it.Ok(); it.Next() {
      r.values[obj.n+it.Index()] = it.Get().ConvertScalar(IntType).(Int)
      r.indexInsert(obj.n+it.Index())
    }
    return r
  }
}
func (v *SparseIntVector) AsMatrix(n, m int) Matrix {
  return v.ToSparseIntMatrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj *SparseIntVector) Dim() int {
  return obj.n
}
func (obj *SparseIntVector) Int8At(i int) int8 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt8()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) Int16At(i int) int16 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt16()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) Int32At(i int) int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt32()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) Int64At(i int) int64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt64()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) IntAt(i int) int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) Float32At(i int) float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat32()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) Float64At(i int) float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat64()
  } else {
    return 0
  }
}
func (obj *SparseIntVector) ConstAt(i int) ConstScalar {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstInt(0.0)
  }
}
func (obj *SparseIntVector) ConstSlice(i, j int) ConstVector {
  return obj.SLICE(i, j)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseIntVector) MapSet(f func(ConstScalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseIntVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseIntVector) ElementType() ScalarType {
  return IntType
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) Permute(pi []int) error {
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
type sortSparseIntVectorByValue struct {
  Value []Int
}
func (obj sortSparseIntVectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseIntVectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseIntVectorByValue) Less(i, j int) bool {
  return obj.Value[i].Smaller(obj.Value[j])
}
func (obj *SparseIntVector) Sort(reverse bool) {
  r := sortSparseIntVectorByValue{}
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
  obj.values = make(map[int]Int)
  obj.vectorSparseIndex = vectorSparseIndex{}
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseIntVectorByValue(r))
  }
  for i := 0; i < len(r.Value); i++ {
    if r.Value[i].GetInt() > 0.0 {
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
func (obj *SparseIntVector) String() string {
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
func (obj *SparseIntVector) Table() string {
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
    buffer.WriteString(fmt.Sprintf("%d:%s", i, ConstInt(0.0)))
  }
  return buffer.String()
}
func (obj *SparseIntVector) Export(filename string) error {
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
func (obj *SparseIntVector) Import(filename string) error {
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
  values := []int{}
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
        values = append(values, int(v))
      }
    }
  }
  *obj = *NewSparseIntVector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []int{}
  r := struct{
    Index []int
    Value []int
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetInt())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseIntVector) UnmarshalJSON(data []byte) error {
  r := struct{
    Index []int
    Value []int
    Length int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  if len(r.Index) != len(r.Value) {
    return fmt.Errorf("invalid sparse vector")
  }
  *obj = *NewSparseIntVector(r.Index, r.Value, r.Length)
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseIntVector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj *SparseIntVector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseIntVector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj *SparseIntVector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseIntVector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseIntVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseIntVector) ITERATOR() *SparseIntVectorIterator {
  r := SparseIntVectorIterator{obj.indexIterator(), obj}
  r.skip()
  return &r
}
func (obj *SparseIntVector) ITERATOR_FROM(i int) *SparseIntVectorIterator {
  r := SparseIntVectorIterator{obj.indexIteratorFrom(i), obj}
  r.skip()
  return &r
}
func (obj *SparseIntVector) JOINT_ITERATOR(b ConstVector) *SparseIntVectorJointIterator {
  r := SparseIntVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, Int{}, nil}
  r.Next()
  return &r
}
func (obj *SparseIntVector) JOINT3_ITERATOR(b, c ConstVector) *SparseIntVectorJoint3Iterator {
  r := SparseIntVectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, Int{}, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseIntVector) JOINT_ITERATOR_(b *SparseIntVector) *SparseIntVectorJointIterator_ {
  r := SparseIntVectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, Int{}, Int{}}
  r.Next()
  return &r
}
func (obj *SparseIntVector) JOINT3_ITERATOR_(b, c *SparseIntVector) *SparseIntVectorJoint3Iterator_ {
  r := SparseIntVectorJoint3Iterator_{obj.ITERATOR(), b.ITERATOR(), c.ITERATOR(), -1, Int{}, Int{}, Int{}}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseIntVectorIterator struct {
  vectorSparseIndexIterator
  v *SparseIntVector
}
func (obj *SparseIntVectorIterator) Get() Scalar {
  if v := obj.GET(); v.ptr == nil {
    return nil
  } else {
    return v
  }
}
func (obj *SparseIntVectorIterator) GetConst() ConstScalar {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return nil
  }
}
func (obj *SparseIntVectorIterator) GET() Int {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return Int{}
  }
}
func (obj *SparseIntVectorIterator) Next() {
  obj.vectorSparseIndexIterator.Next()
  obj.skip()
}
func (obj *SparseIntVectorIterator) skip() {
  for obj.Ok() && obj.GET().nullScalar() {
    i := obj.Index()
    obj.vectorSparseIndexIterator.Next()
    delete(obj.v.values, i)
    obj.v.indexDelete(i)
  }
}
func (obj *SparseIntVectorIterator) Index() int {
  return obj.vectorSparseIndexIterator.Get()
}
func (obj *SparseIntVectorIterator) Clone() *SparseIntVectorIterator {
  return &SparseIntVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseIntVectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseIntVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseIntVectorIterator) CloneIterator() VectorIterator {
  return &SparseIntVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseIntVectorJointIterator struct {
  it1 *SparseIntVectorIterator
  it2 VectorConstIterator
  idx int
  s1 Int
  s2 ConstScalar
}
func (obj *SparseIntVectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseIntVectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt() == int(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt() == int(0))
}
func (obj *SparseIntVectorJointIterator) Next() {
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
func (obj *SparseIntVectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseIntVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseIntVectorJointIterator) GET() (Int, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseIntVectorJointIterator) Clone() *SparseIntVectorJointIterator {
  r := SparseIntVectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *SparseIntVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *SparseIntVectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseIntVectorJoint3Iterator struct {
  it1 *SparseIntVectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 Int
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseIntVectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseIntVectorJoint3Iterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetInt() == int(0)) ||
         !(obj.s2 == nil || obj.s2.GetInt() == int(0)) ||
         !(obj.s3 == nil || obj.s3.GetInt() == int(0))
}
func (obj *SparseIntVectorJoint3Iterator) Next() {
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
    obj.s2 = ConstInt(0.0)
  }
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstInt(0.0)
  }
}
func (obj *SparseIntVectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2, obj.s3
  } else {
    return obj.s1, obj.s2, obj.s3
  }
}
func (obj *SparseIntVectorJoint3Iterator) GET() (Int, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseIntVectorJointIterator_ struct {
  it1 *SparseIntVectorIterator
  it2 *SparseIntVectorIterator
  idx int
  s1 Int
  s2 Int
}
func (obj *SparseIntVectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *SparseIntVectorJointIterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil
}
func (obj *SparseIntVectorJointIterator_) Next() {
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
func (obj *SparseIntVectorJointIterator_) GET() (Int, Int) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseIntVectorJoint3Iterator_ struct {
  it1 *SparseIntVectorIterator
  it2 *SparseIntVectorIterator
  it3 *SparseIntVectorIterator
  idx int
  s1 Int
  s2 Int
  s3 Int
}
func (obj *SparseIntVectorJoint3Iterator_) Index() int {
  return obj.idx
}
func (obj *SparseIntVectorJoint3Iterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil || obj.s3.ptr != nil
}
func (obj *SparseIntVectorJoint3Iterator_) Next() {
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
func (obj *SparseIntVectorJoint3Iterator_) GET() (Int, Int, Int) {
  return obj.s1, obj.s2, obj.s3
}
