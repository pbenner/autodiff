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
/* vector type declaration
 * -------------------------------------------------------------------------- */
type DenseReal32Vector []*Real32
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewDenseReal32Vector(values []float32) DenseReal32Vector {
  v := nilDenseReal32Vector(len(values))
  for i, _ := range values {
    v[i] = NewReal32(values[i])
  }
  return v
}
// Allocate a new vector. All scalars are set to zero.
func NullDenseReal32Vector(length int) DenseReal32Vector {
  v := nilDenseReal32Vector(length)
  if length > 0 {
    for i := 0; i < length; i++ {
      v[i] = NewReal32(0.0)
    }
  }
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilDenseReal32Vector(length int) DenseReal32Vector {
  return make(DenseReal32Vector, length)
}
// Convert vector type.
func AsDenseReal32Vector(v ConstVector) DenseReal32Vector {
  switch v_ := v.(type) {
  case DenseReal32Vector:
    return v_.Clone()
  }
  r := NullDenseReal32Vector(v.Dim())
  for i := 0; i < v.Dim(); i++ {
    r.AT(i).Set(v.ConstAt(i))
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseReal32Vector) Clone() DenseReal32Vector {
  result := make(DenseReal32Vector, len(v))
  for i, _ := range v {
    result[i] = v[i].Clone()
  }
  return result
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) AT(i int) *Real32 {
  return v[i]
}
func (v DenseReal32Vector) SET(w DenseReal32Vector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].SET(w[i])
  }
}
func (v DenseReal32Vector) SLICE(i, j int) DenseReal32Vector {
  return v[i:j]
}
func (v DenseReal32Vector) APPEND(w DenseReal32Vector) DenseReal32Vector {
  return append(v, w...)
}
func (v DenseReal32Vector) ToDenseReal32Matrix(n, m int) *DenseReal32Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseReal32Matrix{}
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
func (v DenseReal32Vector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseReal32Vector) At(i int) Scalar {
  return v.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseReal32Vector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].Set(w.ConstAt(i))
  }
}
func (v DenseReal32Vector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i].Reset()
  }
}
func (v DenseReal32Vector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseReal32Vector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseReal32Vector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseReal32Vector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    switch s := scalar.(type) {
    case *Real32:
      v = append(v, s)
    default:
      v = append(v, s.ConvertScalar(Real32Type).(*Real32))
    }
  }
  return v
}
func (v DenseReal32Vector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case DenseReal32Vector:
    return append(v, w...)
  default:
    for i := 0; i < w.Dim(); i++ {
      v = append(v, w.At(i).ConvertScalar(Real32Type).(*Real32))
    }
    return v
  }
}
func (v DenseReal32Vector) AsMatrix(n, m int) Matrix {
  return v.ToDenseReal32Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseReal32Vector) Dim() int {
  return len(v)
}
func (v DenseReal32Vector) Int8At(i int) int8 {
  return v[i].GetInt8()
}
func (v DenseReal32Vector) Int16At(i int) int16 {
  return v[i].GetInt16()
}
func (v DenseReal32Vector) Int32At(i int) int32 {
  return v[i].GetInt32()
}
func (v DenseReal32Vector) Int64At(i int) int64 {
  return v[i].GetInt64()
}
func (v DenseReal32Vector) IntAt(i int) int {
  return v[i].GetInt()
}
func (v DenseReal32Vector) Float32At(i int) float32 {
  return v[i].GetFloat32()
}
func (v DenseReal32Vector) Float64At(i int) float64 {
  return v[i].GetFloat64()
}
func (v DenseReal32Vector) ConstAt(i int) ConstScalar {
  return v[i]
}
func (v DenseReal32Vector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
func (v DenseReal32Vector) AsConstMatrix(n, m int) ConstMatrix {
  return v.ToDenseReal32Matrix(n, m)
}
/* magic interface
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) CloneMagicVector() MagicVector {
  return v.Clone()
}
func (v DenseReal32Vector) MagicAt(i int) MagicScalar {
  return v.AT(i)
}
func (v DenseReal32Vector) MagicSlice(i, j int) MagicVector {
  return v[i:j]
}
func (v DenseReal32Vector) ResetDerivatives() {
  for i := 0; i < len(v); i++ {
    v[i].ResetDerivatives()
  }
}
func (v DenseReal32Vector) AppendMagicScalar(scalars ...MagicScalar) MagicVector {
  for _, scalar := range scalars {
    switch s := scalar.(type) {
    case *Real32:
      v = append(v, s)
    default:
      v = append(v, s.ConvertMagicScalar(Real32Type).(*Real32))
    }
  }
  return v
}
func (v DenseReal32Vector) AppendMagicVector(w_ MagicVector) MagicVector {
  switch w := w_.(type) {
  case DenseReal32Vector:
    return append(v, w...)
  default:
    for i := 0; i < w.Dim(); i++ {
      v = append(v, w.MagicAt(i).ConvertMagicScalar(Real32Type).(*Real32))
    }
    return v
  }
}
func (v DenseReal32Vector) AsMagicMatrix(n, m int) MagicMatrix {
  return v.ToDenseReal32Matrix(n, m)
}
/* imlement MagicScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f( v[i])
  }
}
func (v DenseReal32Vector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i].Set(f(v.ConstAt(i)))
  }
}
func (v DenseReal32Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseReal32Vector) ElementType() ScalarType {
  return Real32Type
}
func (v DenseReal32Vector) Variables(order int) error {
  for i, _ := range v {
    if err := v[i].SetVariable(i, len(v), order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) Permute(pi []int) error {
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
type sortDenseReal32VectorByValue DenseReal32Vector
func (v sortDenseReal32VectorByValue) Len() int { return len(v) }
func (v sortDenseReal32VectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseReal32VectorByValue) Less(i, j int) bool { return v[i].GetFloat32() < v[j].GetFloat32() }
func (v DenseReal32Vector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseReal32VectorByValue(v)))
  } else {
    sort.Sort(sortDenseReal32VectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseReal32Vector) String() string {
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
func (v DenseReal32Vector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    buffer.WriteString(v[i].String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (v DenseReal32Vector) Export(filename string) error {
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
func (v *DenseReal32Vector) Import(filename string) error {
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
  *v = DenseReal32Vector{}
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
      *v = append(*v, NewReal32(float32(value)))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj DenseReal32Vector) MarshalJSON() ([]byte, error) {
  r := []*Real32{}
  r = obj
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseReal32Vector) UnmarshalJSON(data []byte) error {
  r := []*Real32{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *obj = nilDenseReal32Vector(len(r))
  for i := 0; i < len(r); i++ {
    (*obj)[i] = r[i]
  }
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj DenseReal32Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj DenseReal32Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal32Vector) MagicIterator() VectorMagicIterator {
  return obj.ITERATOR()
}
func (obj DenseReal32Vector) MagicIteratorFrom(i int) VectorMagicIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal32Vector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj DenseReal32Vector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal32Vector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj DenseReal32Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj DenseReal32Vector) ITERATOR() *DenseReal32VectorIterator {
  r := DenseReal32VectorIterator{obj, -1}
  r.Next()
  return &r
}
func (obj DenseReal32Vector) ITERATOR_FROM(i int) *DenseReal32VectorIterator {
  r := DenseReal32VectorIterator{obj, i-1}
  r.Next()
  return &r
}
func (obj DenseReal32Vector) JOINT_ITERATOR(b ConstVector) *DenseReal32VectorJointIterator {
  r := DenseReal32VectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj DenseReal32Vector) JOINT_ITERATOR_(b DenseReal32Vector) *DenseReal32VectorJointIterator_ {
  r := DenseReal32VectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, nil, nil}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type DenseReal32VectorIterator struct {
  v DenseReal32Vector
  i int
}
func (obj *DenseReal32VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseReal32VectorIterator) GetMagic() MagicScalar {
  return obj.GET()
}
func (obj *DenseReal32VectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseReal32VectorIterator) GET() *Real32 {
  return obj.v[obj.i]
}
func (obj *DenseReal32VectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseReal32VectorIterator) Next() {
  obj.i++
}
func (obj *DenseReal32VectorIterator) Index() int {
  return obj.i
}
func (obj *DenseReal32VectorIterator) Clone() *DenseReal32VectorIterator {
  return &DenseReal32VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal32VectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseReal32VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal32VectorIterator) CloneMagicIterator() VectorMagicIterator {
  return &DenseReal32VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal32VectorIterator) CloneIterator() VectorIterator {
  return &DenseReal32VectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal32VectorJointIterator struct {
  it1 *DenseReal32VectorIterator
  it2 VectorConstIterator
  idx int
  s1 *Real32
  s2 ConstScalar
}
func (obj *DenseReal32VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseReal32VectorJointIterator) Ok() bool {
  return !(obj.s1 == nil || obj.s1.GetFloat32() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat32() == 0.0)
}
func (obj *DenseReal32VectorJointIterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = nil
  obj.s2 = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1 = nil
      obj.s2 = obj.it2.GetConst()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GetConst()
    }
  }
  if obj.s1 != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstFloat32(0.0)
  }
}
func (obj *DenseReal32VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal32VectorJointIterator) GetMagic() (MagicScalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal32VectorJointIterator) Get() (Scalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal32VectorJointIterator) GET() (*Real32, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal32VectorJointIterator) Clone() *DenseReal32VectorJointIterator {
  r := DenseReal32VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseReal32VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *DenseReal32VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal32VectorJointIterator_ struct {
  it1 *DenseReal32VectorIterator
  it2 *DenseReal32VectorIterator
  idx int
  s1 *Real32
  s2 *Real32
}
func (obj *DenseReal32VectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *DenseReal32VectorJointIterator_) Ok() bool {
  return obj.s1 != nil || obj.s2 != nil
}
func (obj *DenseReal32VectorJointIterator_) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  obj.s1 = nil
  obj.s2 = nil
  if ok1 {
    obj.idx = obj.it1.Index()
    obj.s1 = obj.it1.GET()
  }
  if ok2 {
    switch {
    case obj.idx > obj.it2.Index() || !ok1:
      obj.idx = obj.it2.Index()
      obj.s1 = nil
      obj.s2 = obj.it2.GET()
    case obj.idx == obj.it2.Index():
      obj.s2 = obj.it2.GET()
    }
  }
  if obj.s1 != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  }
}
func (obj *DenseReal32VectorJointIterator_) GET() (*Real32, *Real32) {
  return obj.s1, obj.s2
}
