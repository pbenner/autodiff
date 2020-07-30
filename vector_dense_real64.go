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
type DenseReal64Vector []*Real64
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewDenseReal64Vector(values []float64) DenseReal64Vector {
  v := nilDenseReal64Vector(len(values))
  for i, _ := range values {
    v[i] = NewReal64(values[i])
  }
  return v
}
// Allocate a new vector. All scalars are set to zero.
func NullDenseReal64Vector(length int) DenseReal64Vector {
  v := nilDenseReal64Vector(length)
  if length > 0 {
    for i := 0; i < length; i++ {
      v[i] = NewReal64(0.0)
    }
  }
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilDenseReal64Vector(length int) DenseReal64Vector {
  return make(DenseReal64Vector, length)
}
// Convert vector type.
func AsDenseReal64Vector(v ConstVector) DenseReal64Vector {
  switch v_ := v.(type) {
  case DenseReal64Vector:
    return v_.Clone()
  }
  r := NullDenseReal64Vector(v.Dim())
  for i := 0; i < v.Dim(); i++ {
    r.AT(i).Set(v.ConstAt(i))
  }
  return r
}
/* cloning
 * -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseReal64Vector) Clone() DenseReal64Vector {
  result := make(DenseReal64Vector, len(v))
  for i, _ := range v {
    result[i] = v[i].Clone()
  }
  return result
}
/* native vector methods
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) AT(i int) *Real64 {
  return v[i]
}
func (v DenseReal64Vector) SET(w DenseReal64Vector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].SET(w[i])
  }
}
func (v DenseReal64Vector) SLICE(i, j int) DenseReal64Vector {
  return v[i:j]
}
func (v DenseReal64Vector) APPEND(w DenseReal64Vector) DenseReal64Vector {
  return append(v, w...)
}
func (v DenseReal64Vector) ToDenseReal64Matrix(n, m int) *DenseReal64Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  matrix := DenseReal64Matrix{}
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
func (v DenseReal64Vector) CloneVector() Vector {
  return v.Clone()
}
func (v DenseReal64Vector) At(i int) Scalar {
  return v.AT(i)
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseReal64Vector) Set(w ConstVector) {
  if v.Dim() != w.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for i := 0; i < w.Dim(); i++ {
    v[i].Set(w.ConstAt(i))
  }
}
func (v DenseReal64Vector) Reset() {
  for i := 0; i < len(v); i++ {
    v[i].Reset()
  }
}
func (v DenseReal64Vector) ReverseOrder() {
  n := len(v)
  for i := 0; i < n/2; i++ {
    v[i], v[n-1-i] = v[n-1-i], v[i]
  }
}
func (v DenseReal64Vector) Slice(i, j int) Vector {
  return v[i:j]
}
func (v DenseReal64Vector) Swap(i, j int) {
  v[i], v[j] = v[j], v[i]
}
func (v DenseReal64Vector) AppendScalar(scalars ...Scalar) Vector {
  for _, scalar := range scalars {
    switch s := scalar.(type) {
    case *Real64:
      v = append(v, s)
    default:
      v = append(v, s.ConvertScalar(Real64Type).(*Real64))
    }
  }
  return v
}
func (v DenseReal64Vector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case DenseReal64Vector:
    return append(v, w...)
  default:
    for i := 0; i < w.Dim(); i++ {
      v = append(v, w.At(i).ConvertScalar(Real64Type).(*Real64))
    }
    return v
  }
}
func (v DenseReal64Vector) AsMatrix(n, m int) Matrix {
  return v.ToDenseReal64Matrix(n, m)
}
/* const interface
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) CloneConstVector() ConstVector {
  return v.Clone()
}
func (v DenseReal64Vector) Dim() int {
  return len(v)
}
func (v DenseReal64Vector) Int8At(i int) int8 {
  return v[i].GetInt8()
}
func (v DenseReal64Vector) Int16At(i int) int16 {
  return v[i].GetInt16()
}
func (v DenseReal64Vector) Int32At(i int) int32 {
  return v[i].GetInt32()
}
func (v DenseReal64Vector) Int64At(i int) int64 {
  return v[i].GetInt64()
}
func (v DenseReal64Vector) IntAt(i int) int {
  return v[i].GetInt()
}
func (v DenseReal64Vector) Float32At(i int) float32 {
  return v[i].GetFloat32()
}
func (v DenseReal64Vector) Float64At(i int) float64 {
  return v[i].GetFloat64()
}
func (v DenseReal64Vector) ConstAt(i int) ConstScalar {
  return v[i]
}
func (v DenseReal64Vector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}
func (v DenseReal64Vector) AsConstMatrix(n, m int) ConstMatrix {
  return v.ToDenseReal64Matrix(n, m)
}
/* magic interface
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) CloneMagicVector() MagicVector {
  return v.Clone()
}
func (v DenseReal64Vector) MagicAt(i int) MagicScalar {
  return v.AT(i)
}
func (v DenseReal64Vector) MagicSlice(i, j int) MagicVector {
  return v[i:j]
}
func (v DenseReal64Vector) ResetDerivatives() {
  for i := 0; i < len(v); i++ {
    v[i].ResetDerivatives()
  }
}
func (v DenseReal64Vector) AppendMagicScalar(scalars ...MagicScalar) MagicVector {
  for _, scalar := range scalars {
    switch s := scalar.(type) {
    case *Real64:
      v = append(v, s)
    default:
      v = append(v, s.ConvertMagicScalar(Real64Type).(*Real64))
    }
  }
  return v
}
func (v DenseReal64Vector) AppendMagicVector(w_ MagicVector) MagicVector {
  switch w := w_.(type) {
  case DenseReal64Vector:
    return append(v, w...)
  default:
    for i := 0; i < w.Dim(); i++ {
      v = append(v, w.MagicAt(i).ConvertMagicScalar(Real64Type).(*Real64))
    }
    return v
  }
}
func (v DenseReal64Vector) AsMagicMatrix(n, m int) MagicMatrix {
  return v.ToDenseReal64Matrix(n, m)
}
/* imlement MagicScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) Map(f func(Scalar)) {
  for i := 0; i < len(v); i++ {
    f( v[i])
  }
}
func (v DenseReal64Vector) MapSet(f func(ConstScalar) Scalar) {
  for i := 0; i < len(v); i++ {
    v[i].Set(f(v.ConstAt(i)))
  }
}
func (v DenseReal64Vector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < len(v); i++ {
    r = f(r, v.ConstAt(i))
  }
  return r
}
func (v DenseReal64Vector) ElementType() ScalarType {
  return Real64Type
}
func (v DenseReal64Vector) Variables(order int) error {
  for i, _ := range v {
    if err := v[i].SetVariable(i, len(v), order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) Permute(pi []int) error {
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
type sortDenseReal64VectorByValue DenseReal64Vector
func (v sortDenseReal64VectorByValue) Len() int { return len(v) }
func (v sortDenseReal64VectorByValue) Swap(i, j int) { v[i], v[j] = v[j], v[i] }
func (v sortDenseReal64VectorByValue) Less(i, j int) bool { return v[i].GetFloat64() < v[j].GetFloat64() }
func (v DenseReal64Vector) Sort(reverse bool) {
  if reverse {
    sort.Sort(sort.Reverse(sortDenseReal64VectorByValue(v)))
  } else {
    sort.Sort(sortDenseReal64VectorByValue(v))
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseReal64Vector) String() string {
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
func (v DenseReal64Vector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    buffer.WriteString(v[i].String())
    buffer.WriteString("\n")
  }
  return buffer.String()
}
func (v DenseReal64Vector) Export(filename string) error {
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
func (v *DenseReal64Vector) Import(filename string) error {
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
  *v = DenseReal64Vector{}
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
      *v = append(*v, NewReal64(float64(value)))
    }
  }
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj DenseReal64Vector) MarshalJSON() ([]byte, error) {
  r := []*Real64{}
  r = obj
  return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseReal64Vector) UnmarshalJSON(data []byte) error {
  r := []*Real64{}
  if err := json.Unmarshal(data, &r); err != nil {
    return err
  }
  *obj = nilDenseReal64Vector(len(r))
  for i := 0; i < len(r); i++ {
    (*obj)[i] = r[i]
  }
  return nil
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj DenseReal64Vector) ConstIterator() VectorConstIterator {
  return obj.ITERATOR()
}
func (obj DenseReal64Vector) ConstIteratorFrom(i int) VectorConstIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal64Vector) MagicIterator() VectorMagicIterator {
  return obj.ITERATOR()
}
func (obj DenseReal64Vector) MagicIteratorFrom(i int) VectorMagicIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal64Vector) Iterator() VectorIterator {
  return obj.ITERATOR()
}
func (obj DenseReal64Vector) IteratorFrom(i int) VectorIterator {
  return obj.ITERATOR_FROM(i)
}
func (obj DenseReal64Vector) JointIterator(b ConstVector) VectorJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj DenseReal64Vector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  return obj.JOINT_ITERATOR(b)
}
func (obj DenseReal64Vector) ITERATOR() *DenseReal64VectorIterator {
  r := DenseReal64VectorIterator{obj, -1}
  r.Next()
  return &r
}
func (obj DenseReal64Vector) ITERATOR_FROM(i int) *DenseReal64VectorIterator {
  r := DenseReal64VectorIterator{obj, i-1}
  r.Next()
  return &r
}
func (obj DenseReal64Vector) JOINT_ITERATOR(b ConstVector) *DenseReal64VectorJointIterator {
  r := DenseReal64VectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj DenseReal64Vector) JOINT_ITERATOR_(b DenseReal64Vector) *DenseReal64VectorJointIterator_ {
  r := DenseReal64VectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, nil, nil}
  r.Next()
  return &r
}
/* iterator
 * -------------------------------------------------------------------------- */
type DenseReal64VectorIterator struct {
  v DenseReal64Vector
  i int
}
func (obj *DenseReal64VectorIterator) GetConst() ConstScalar {
  return obj.GET()
}
func (obj *DenseReal64VectorIterator) GetMagic() MagicScalar {
  return obj.GET()
}
func (obj *DenseReal64VectorIterator) Get() Scalar {
  return obj.GET()
}
func (obj *DenseReal64VectorIterator) GET() *Real64 {
  return obj.v[obj.i]
}
func (obj *DenseReal64VectorIterator) Ok() bool {
  return obj.i < len(obj.v)
}
func (obj *DenseReal64VectorIterator) Next() {
  obj.i++
}
func (obj *DenseReal64VectorIterator) Index() int {
  return obj.i
}
func (obj *DenseReal64VectorIterator) Clone() *DenseReal64VectorIterator {
  return &DenseReal64VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal64VectorIterator) CloneConstIterator() VectorConstIterator {
  return &DenseReal64VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal64VectorIterator) CloneMagicIterator() VectorMagicIterator {
  return &DenseReal64VectorIterator{obj.v, obj.i}
}
func (obj *DenseReal64VectorIterator) CloneIterator() VectorIterator {
  return &DenseReal64VectorIterator{obj.v, obj.i}
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal64VectorJointIterator struct {
  it1 *DenseReal64VectorIterator
  it2 VectorConstIterator
  idx int
  s1 *Real64
  s2 ConstScalar
}
func (obj *DenseReal64VectorJointIterator) Index() int {
  return obj.idx
}
func (obj *DenseReal64VectorJointIterator) Ok() bool {
  return !(obj.s1 == nil || obj.s1.GetFloat64() == 0.0) ||
         !(obj.s2 == nil || obj.s2.GetFloat64() == 0.0)
}
func (obj *DenseReal64VectorJointIterator) Next() {
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
    obj.s2 = ConstFloat64(0.0)
  }
}
func (obj *DenseReal64VectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal64VectorJointIterator) GetMagic() (MagicScalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal64VectorJointIterator) Get() (Scalar, ConstScalar) {
  return obj.GET()
}
func (obj *DenseReal64VectorJointIterator) GET() (*Real64, ConstScalar) {
  if obj.s1 == nil {
    return nil, obj.s2
  } else {
    return obj.s1, obj.s2
  }
}
func (obj *DenseReal64VectorJointIterator) Clone() *DenseReal64VectorJointIterator {
  r := DenseReal64VectorJointIterator{}
  r.it1 = obj.it1.Clone()
  r.it2 = obj.it2.CloneConstIterator()
  r.idx = obj.idx
  r.s1 = obj.s1
  r.s2 = obj.s2
  return &r
}
func (obj *DenseReal64VectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
  return obj.Clone()
}
func (obj *DenseReal64VectorJointIterator) CloneJointIterator() VectorJointIterator {
  return obj.Clone()
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseReal64VectorJointIterator_ struct {
  it1 *DenseReal64VectorIterator
  it2 *DenseReal64VectorIterator
  idx int
  s1 *Real64
  s2 *Real64
}
func (obj *DenseReal64VectorJointIterator_) Index() int {
  return obj.idx
}
func (obj *DenseReal64VectorJointIterator_) Ok() bool {
  return obj.s1 != nil || obj.s2 != nil
}
func (obj *DenseReal64VectorJointIterator_) Next() {
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
func (obj *DenseReal64VectorJointIterator_) GET() (*Real64, *Real64) {
  return obj.s1, obj.s2
}
