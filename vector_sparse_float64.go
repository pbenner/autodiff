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
type SparseFloat64Vector struct {
  vectorSparseIndex
  values map[int]Float64
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseFloat64Vector(indices []int, values []float64, n int) *SparseFloat64Vector {
  if len(indices) != len(values) {
    panic("number of indices does not match number of values")
  }
  r := nilSparseFloat64Vector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if _, ok := r.values[k]; ok {
      panic("index appeared multiple times")
    } else {
      if values[i] != 0.0 {
        r.values[k] = NewFloat64(values[i])
        r.indexInsert(k)
      }
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseFloat64Vector(length int) *SparseFloat64Vector {
  v := nilSparseFloat64Vector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseFloat64Vector(length int) *SparseFloat64Vector {
  return &SparseFloat64Vector{values: make(map[int]Float64), n: length}
}
// Convert vector type.
func AsSparseFloat64Vector(v ConstVector) *SparseFloat64Vector {
  switch v_ := v.(type) {
  case *SparseFloat64Vector:
    return v_.Clone()
  }
  r := NullSparseFloat64Vector(v.Dim())
  for it := v.ConstIterator(); it.Ok(); it.Next() {
    r.AT(it.Index()).Set(it.GetConst())
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseFloat64Vector) Clone() *SparseFloat64Vector {
  r := nilSparseFloat64Vector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndex = obj.indexClone()
  return r
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) AT(i int) Float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullFloat64()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseFloat64Vector) AT_(i int) Float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return Float64{}
  }
}
func (obj *SparseFloat64Vector) SET(x *SparseFloat64Vector) {
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
    case s1.ptr != nil : s1.SetFloat64(0)
    default : obj.AT(it.Index()).SET(s2)
    }
  }
}
func (obj *SparseFloat64Vector) SLICE(i, j int) *SparseFloat64Vector {
  r := nilSparseFloat64Vector(j-i)
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
func (obj *SparseFloat64Vector) APPEND(w *SparseFloat64Vector) *SparseFloat64Vector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for it := w.ITERATOR(); it.Ok(); it.Next() {
    i := obj.n+it.Index()
    r.values[i] = it.GET()
    r.indexInsert(i)
  }
  return r
}
func (obj *SparseFloat64Vector) ToSparseFloat64Matrix(n, m int) *SparseFloat64Matrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullSparseFloat64Vector(obj.n)
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    v.At(it.Index()).Set(it.GET())
  }
  matrix := SparseFloat64Matrix{}
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
func (obj *SparseFloat64Vector) CloneVector() Vector {
  return obj.Clone()
}
func (obj *SparseFloat64Vector) At(i int) Scalar {
  return obj.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseFloat64Vector) Set(x ConstVector) {
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
    case s1.ptr != nil : s1.SetFloat64(0)
    default : obj.AT(it.Index()).Set(s2)
    }
  }
}
func (obj *SparseFloat64Vector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseFloat64Vector) ReverseOrder() {
  n := obj.Dim()
  values := make(map[int]Float64)
  index := vectorSparseIndex{}
  for i, s := range obj.values {
    j := n-i-1
    values[j] = s
    index.indexInsert(j)
  }
  obj.values = values
  obj.vectorSparseIndex = index
}
func (obj *SparseFloat64Vector) Slice(i, j int) Vector {
  return obj.SLICE(i, j)
}
func (obj *SparseFloat64Vector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
}
func (obj *SparseFloat64Vector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case Float64:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertScalar(Float64Type).(Float64)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseFloat64Vector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseFloat64Vector:
    return obj.APPEND(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for it := w.Iterator(); it.Ok(); it.Next() {
      r.values[obj.n+it.Index()] = it.Get().ConvertScalar(Float64Type).(Float64)
      r.indexInsert(obj.n+it.Index())
    }
    return r
  }
}
func (v *SparseFloat64Vector) AsMatrix(n, m int) Matrix {
  return v.ToSparseFloat64Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) CloneConstVector() ConstVector {
  return obj.Clone()
}
func (obj *SparseFloat64Vector) Dim() int {
  return obj.n
}
func (obj *SparseFloat64Vector) Int8At(i int) int8 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt8()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) Int16At(i int) int16 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt16()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) Int32At(i int) int32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt32()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) Int64At(i int) int64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt64()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) IntAt(i int) int {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetInt()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) Float32At(i int) float32 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat32()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) Float64At(i int) float64 {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v.GetFloat64()
  } else {
    return 0
  }
}
func (obj *SparseFloat64Vector) ConstAt(i int) ConstScalar {
  if i < 0 || i >= obj.Dim() {
    panic("index out of bounds")
  }
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstFloat64(0.0)
  }
}
func (obj *SparseFloat64Vector) ConstSlice(i, j int) ConstVector {
  return obj.SLICE(i, j)
}
func (v *SparseFloat64Vector) AsConstMatrix(n, m int) ConstMatrix {
  return v.ToSparseFloat64Matrix(n, m)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseFloat64Vector) MapSet(f func(ConstScalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseFloat64Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseFloat64Vector) ElementType() ScalarType {
  return Float64Type
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) Permute(pi []int) error {
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
type sortSparseFloat64VectorByValue struct {
  Value []Float64
}
func (obj sortSparseFloat64VectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseFloat64VectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseFloat64VectorByValue) Less(i, j int) bool {
  return obj.Value[i].Smaller(obj.Value[j])
}
func (obj *SparseFloat64Vector) Sort(reverse bool) {
  r := sortSparseFloat64VectorByValue{}
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
  obj.values = make(map[int]Float64)
  obj.vectorSparseIndex = vectorSparseIndex{}
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseFloat64VectorByValue(r))
  }
  for i := 0; i < len(r.Value); i++ {
    if r.Value[i].GetFloat64() > 0.0 {
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
func (obj *SparseFloat64Vector) String() string {
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
func (obj *SparseFloat64Vector) Table() string {
  var buffer bytes.Buffer
  n := obj.Dim()
  for i := 0; i < n; i++ {
    buffer.WriteString(obj.ConstAt(i).String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (obj *SparseFloat64Vector) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()
  w := bufio.NewWriter(f)
  defer w.Flush()
  if _, err := fmt.Fprintf(w, "%d\n", obj.Dim()); err != nil {
    return err
  }
  for it := obj.ITERATOR(); it.Ok(); it.Next() {
    i := it.Index()
    if _, err := fmt.Fprintf(w, "%d %v\n", i, it.GET()); err != nil {
      return err
    }
  }
  return nil
}
func (obj *SparseFloat64Vector) Import(filename string) error {
  values := []float64{}
  indices := []int{}
  n := 0
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
  // scan header
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
    if len(fields) != 1 {
      return fmt.Errorf("invalid sparse vector format")
    }
    if v, err := strconv.ParseInt(fields[0], 10, 64); err != nil {
      return err
    } else {
      n = int(v)
    }
    break
  }
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
    if len(fields) != 2 {
      return fmt.Errorf("invalid sparse vector format")
    }
    if v, err := strconv.ParseInt(fields[0], 10, 64); err != nil {
      return err
    } else {
      indices = append(indices, int(v))
    }
    if v, err := strconv.ParseFloat(fields[1], 64); err != nil {
      return err
    } else {
      values = append(values, float64(v))
    }
  }
  *obj = *NewSparseFloat64Vector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []float64{}
  r := struct{
    Index []int
    Value []float64
    Length int}{}
  for it := obj.ConstIterator(); it.Ok(); it.Next() {
    k = append(k, it.Index())
    v = append(v, it.GetConst().GetFloat64())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseFloat64Vector) UnmarshalJSON(data []byte) error {
  r := struct{
    Index []int
    Value []float64
    Length int}{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  if len(r.Index) != len(r.Value) {
    return fmt.Errorf("invalid sparse vector")
  }
  *obj = *NewSparseFloat64Vector(r.Index, r.Value, r.Length)
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseFloat64Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj *SparseFloat64Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseFloat64Vector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj *SparseFloat64Vector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj *SparseFloat64Vector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseFloat64Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj *SparseFloat64Vector) ITERATOR() *SparseFloat64VectorIterator {
  r := SparseFloat64VectorIterator{obj.indexIterator(), obj}
  r.skip()
  return &r
}
func (obj *SparseFloat64Vector) ITERATOR_FROM(i int) *SparseFloat64VectorIterator {
  r := SparseFloat64VectorIterator{obj.indexIteratorFrom(i), obj}
  r.skip()
  return &r
}
func (obj *SparseFloat64Vector) JOINT_ITERATOR(b ConstVector) *SparseFloat64VectorJointIterator {
  r := SparseFloat64VectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, Float64{}, nil}
  r.Next()
  return &r
}
func (obj *SparseFloat64Vector) JOINT3_ITERATOR(b, c ConstVector) *SparseFloat64VectorJoint3Iterator {
  r := SparseFloat64VectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, Float64{}, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseFloat64Vector) JOINT_ITERATOR_(b *SparseFloat64Vector) *SparseFloat64VectorJointIterator_ {
  r := SparseFloat64VectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, Float64{}, Float64{}}
  r.Next()
  return &r
}
func (obj *SparseFloat64Vector) JOINT3_ITERATOR_(b, c *SparseFloat64Vector) *SparseFloat64VectorJoint3Iterator_ {
  r := SparseFloat64VectorJoint3Iterator_{obj.ITERATOR(), b.ITERATOR(), c.ITERATOR(), -1, Float64{}, Float64{}, Float64{}}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseFloat64VectorIterator struct {
  vectorSparseIndexIterator
  v *SparseFloat64Vector
}
func (obj *SparseFloat64VectorIterator) Get() Scalar {
  if v := obj.GET(); v.ptr == nil {
    return nil
  } else {
    return v
  }
}
func (obj *SparseFloat64VectorIterator) GetConst() ConstScalar {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return nil
  }
}
func (obj *SparseFloat64VectorIterator) GET() Float64 {
  if v, ok := obj.v.values[obj.Index()]; ok {
    return v
  } else {
    return Float64{}
  }
}
func (obj *SparseFloat64VectorIterator) Next() {
  obj.vectorSparseIndexIterator.Next()
  obj.skip()
}
func (obj *SparseFloat64VectorIterator) skip() {
  for obj.Ok() && obj.GET().nullScalar() {
    i := obj.Index()
    obj.vectorSparseIndexIterator.Next()
    delete(obj.v.values, i)
    obj.v.indexDelete(i)
  }
}
func (obj *SparseFloat64VectorIterator) Index() int {
  return obj.vectorSparseIndexIterator.Get()
}
func (obj *SparseFloat64VectorIterator) Clone() *SparseFloat64VectorIterator {
  return &SparseFloat64VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseFloat64VectorIterator) CloneConstIterator() VectorConstIterator {
  return &SparseFloat64VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseFloat64VectorIterator) CloneIterator() VectorIterator {
  return &SparseFloat64VectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat64VectorJointIterator struct {
  it1 *SparseFloat64VectorIterator
  it2 VectorConstIterator
  idx int
  s1 Float64
  s2 ConstScalar
}
func (obj *SparseFloat64VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseFloat64VectorJointIterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == float64(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == float64(0))
}
func (obj *SparseFloat64VectorJointIterator) Next() {
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
func (obj *SparseFloat64VectorJointIterator) Get() (Scalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseFloat64VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *SparseFloat64VectorJointIterator) GET() (Float64, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseFloat64VectorJointIterator) Clone() *SparseFloat64VectorJointIterator {
  r := SparseFloat64VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *SparseFloat64VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *SparseFloat64VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat64VectorJoint3Iterator struct {
  it1 *SparseFloat64VectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 Float64
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseFloat64VectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseFloat64VectorJoint3Iterator) Ok() bool {
  return !(obj.s1.ptr == nil || obj.s1.GetFloat64() == float64(0)) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == float64(0)) ||
         !(obj.s3 == nil || obj.s3.GetFloat64() == float64(0))
}
func (obj *SparseFloat64VectorJoint3Iterator) Next() {
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
    obj.s2 = ConstFloat64(0.0)
  }
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstFloat64(0.0)
  }
}
func (obj *SparseFloat64VectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  if obj.s1.ptr == nil {
    return nil, obj.s2, obj.s3
  } else {
    return obj.s1, obj.s2, obj.s3
  }
}
func (obj *SparseFloat64VectorJoint3Iterator) GET() (Float64, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat64VectorJointIterator_ struct {
  it1 *SparseFloat64VectorIterator
  it2 *SparseFloat64VectorIterator
  idx int
  s1 Float64
  s2 Float64
}
func (obj *SparseFloat64VectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *SparseFloat64VectorJointIterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil
}
func (obj *SparseFloat64VectorJointIterator_) Next() {
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
func (obj *SparseFloat64VectorJointIterator_) GET() (Float64, Float64) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseFloat64VectorJoint3Iterator_ struct {
  it1 *SparseFloat64VectorIterator
  it2 *SparseFloat64VectorIterator
  it3 *SparseFloat64VectorIterator
  idx int
  s1 Float64
  s2 Float64
  s3 Float64
}
func (obj *SparseFloat64VectorJoint3Iterator_) Index() int {
  return obj.idx
}
func (obj *SparseFloat64VectorJoint3Iterator_) Ok() bool {
  return obj.s1.ptr != nil || obj.s2.ptr != nil || obj.s3.ptr != nil
}
func (obj *SparseFloat64VectorJoint3Iterator_) Next() {
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
func (obj *SparseFloat64VectorJoint3Iterator_) GET() (Float64, Float64, Float64) {
  return obj.s1, obj.s2, obj.s3
}
