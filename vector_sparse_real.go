/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2019 Philipp Benner
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
package autodiff
/* -------------------------------------------------------------------------- */
import "fmt"
import "bytes"
import "bufio"
import "encoding/json"
import "errors"
import "compress/gzip"
import "sort"
import "strconv"
import "strings"
import "os"
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* vector type declaration
 * -------------------------------------------------------------------------- */
type SparseRealVector struct {
  vectorSparseIndexSlice
  values map[int]*Real
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseRealVector(indices []int, values []float64, n int) *SparseRealVector {
  r := nilSparseRealVector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if values[i] != 0.0 {
      r.values[k] = NewReal(values[i])
      r.indexInsert(k)
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseRealVector(length int) *SparseRealVector {
  v := nilSparseRealVector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseRealVector(length int) *SparseRealVector {
  return &SparseRealVector{values: make(map[int]*Real), n: length}
}
// Convert vector type.
func AsSparseRealVector(v ConstVector) *SparseRealVector {
  switch v_ := v.(type) {
  case *SparseRealVector:
    return v_
  }
  r := NullSparseRealVector(v.Dim())
  for entry := range v.ConstRange() {
    r.AT(entry.Index).Set(entry.Value)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseRealVector) Clone() *SparseRealVector {
  r := nilSparseRealVector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndexSlice = obj.indexClone()
  return r
}
func (obj *SparseRealVector) CloneVector() Vector {
  return obj.Clone()
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseRealVector) Set(x ConstVector) {
  if obj.Dim() != x.Dim() {
    panic("Set(): Vector dimensions do not match!")
  }
  for entry := range obj.JOINT_RANGE(x) {
    switch {
    case entry.Value1 != nil && entry.Value2 != nil: entry.Value1.Set(entry.Value2)
    case entry.Value1 != nil : entry.Value1.SetValue(0.0)
    default : obj.AT(entry.Index).Set(entry.Value2)
    }
  }
}
/* const vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) ValueAt(i int) float64 {
  if v, ok := obj.values[i]; ok {
    return v.GetValue()
  } else {
    return 0.0
  }
}
func (obj *SparseRealVector) ConstAt(i int) ConstScalar {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstReal(0.0)
  }
}
func (obj *SparseRealVector) ConstSlice(i, j int) ConstVector {
  return obj.Slice(i, j)
}
func (obj *SparseRealVector) GetValues() []float64 {
  r := make([]float64, obj.Dim())
  for i, v := range obj.values {
    r[i] = v.GetValue()
  }
  return r
}
func (obj *SparseRealVector) ConstRange() chan VectorConstRangeType {
  channel := make(chan VectorConstRangeType)
  go func() {
    obj.indexSort()
    for k, i := range obj.index {
      if s := obj.values[i]; obj.nullScalar(s) {
        obj.indexRevoke(k)
        delete(obj.values, i)
      } else {
        channel <- VectorConstRangeType{i, s}
      }
    }
    close(channel)
  }()
  return channel
}
/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) ConstIterator() VectorConstIterator {
  r := SparseRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseRealVector) JointIterator(b ConstVector) VectorJointIterator {
  r := SparseRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  r := SparseRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseRealVector) Iterator() VectorIterator {
  r := SparseRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseRealVector) ITERATOR() *SparseRealVectorIterator {
  r := SparseRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseRealVector) CONST_JOINT_ITERATOR(b ConstVector) *SparseRealVectorJointIterator {
  r := SparseRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseRealVector) CONST_JOINT3_ITERATOR(b, c ConstVector) *SparseRealVectorJoint3Iterator {
  r := SparseRealVectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, nil, nil, nil}
  r.Next()
  return &r
}
/* range methods
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) Range() chan VectorRangeType {
  channel := make(chan VectorRangeType)
  go func() {
    obj.indexSort()
    for k, i := range obj.index {
      if s := obj.values[i]; obj.nullScalar(s) {
        obj.indexRevoke(k)
        delete(obj.values, i)
      } else {
        channel <- VectorRangeType{i, s}
      }
    }
    close(channel)
  }()
  return channel
}
func (obj *SparseRealVector) JointRange(b ConstVector) chan VectorJointRangeType {
  channel := make(chan VectorJointRangeType)
  go func() {
    c1 := obj. RANGE()
    c2 := b.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    for ok1 || ok2 {
      r := VectorJointRangeType{}
      r.Value2 = ConstReal(0.0)
      if ok1 {
        r.Index = r1.Index
        r.Value1 = r1.Value
      }
      if ok2 {
        switch {
        case r.Index > r2.Index || !ok1:
          r.Index = r2.Index
          r.Value1 = nil
          r.Value2 = r2.Value
        case r.Index == r2.Index:
          r.Value2 = r2.Value
        }
      }
      channel <- r
    }
    close(channel)
  }()
  return channel
}
type SparseRealVectorRange struct {
  Index int
  Value *Real
}
func (obj *SparseRealVector) RANGE() chan SparseRealVectorRange {
  channel := make(chan SparseRealVectorRange)
  go func() {
    obj.indexSort()
    for k, i := range obj.index {
      if s := obj.values[i]; obj.nullScalar(s) {
        obj.indexRevoke(k)
        delete(obj.values, i)
      } else {
        channel <- SparseRealVectorRange{i, s}
      }
    }
    close(channel)
  }()
  return channel
}
type SparseRealVectorJointRange struct {
  Index int
  Value1 *Real
  Value2 ConstScalar
}
func (obj *SparseRealVector) JOINT_RANGE(b ConstVector) chan SparseRealVectorJointRange {
  channel := make(chan SparseRealVectorJointRange)
  go func() {
    c1 := obj. RANGE()
    c2 := b.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    for ok1 || ok2 {
      r := SparseRealVectorJointRange{}
      if ok1 {
        r.Index = r1.Index
        r.Value1 = r1.Value
      }
      if ok2 {
        switch {
        case r.Index > r2.Index || !ok1:
          r.Index = r2.Index
          r.Value1 = nil
          r.Value2 = r2.Value
        case r.Index == r2.Index:
          r.Value2 = r2.Value
        }
      }
      if r.Value1 != nil {
        r1, ok1 = <- c1
      }
      if r.Value2 != nil {
        r2, ok2 = <- c2
      } else {
        r.Value2 = ConstReal(0.0)
      }
      channel <- r
    }
    close(channel)
  }()
  return channel
}
type SparseRealVectorJointRange3 struct {
  Index int
  Value1 *Real
  Value2 ConstScalar
  Value3 ConstScalar
}
func (obj *SparseRealVector) JOINT_RANGE3(b, c ConstVector) chan SparseRealVectorJointRange3 {
  channel := make(chan SparseRealVectorJointRange3)
  go func() {
    c1 := obj. RANGE()
    c2 := b.ConstRange()
    c3 := c.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    r3, ok3 := <- c3
    for ok1 || ok2 || ok3 {
      r := SparseRealVectorJointRange3{}
      if ok1 {
        r.Index = r1.Index
        r.Value1 = r1.Value
      }
      if ok2 {
        switch {
        case r.Index > r2.Index || !ok1:
          r.Index = r2.Index
          r.Value1 = nil
          r.Value2 = r2.Value
        case r.Index == r2.Index:
          r.Value2 = r2.Value
        }
      }
      if ok3 {
        switch {
        case r.Index > r3.Index || (!ok1 && !ok2):
          r.Index = r3.Index
          r.Value1 = nil
          r.Value2 = nil
          r.Value3 = r3.Value
        case r.Index == r3.Index:
          r.Value3 = r3.Value
        }
      }
      if r.Value1 != nil {
        r1, ok1 = <- c1
      }
      if r.Value2 != nil {
        r2, ok2 = <- c2
      } else {
        r.Value2 = ConstReal(0.0)
      }
      if r.Value3 != nil {
        r3, ok3 = <- c3
      } else {
        r.Value3 = ConstReal(0.0)
      }
      channel <- r
    }
    close(channel)
  }()
  return channel
}
/* -------------------------------------------------------------------------- */
func (obj *SparseRealVector) Dim() int {
  return obj.n
}
func (obj *SparseRealVector) At(i int) Scalar {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullReal()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseRealVector) AT(i int) *Real {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullReal()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseRealVector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseRealVector) ResetDerivatives() {
  for _, v := range obj.values {
    v.ResetDerivatives()
  }
}
func (obj *SparseRealVector) ReverseOrder() {
  n := obj.Dim()
  v := make(map[int]*Real)
  for i, s := range obj.values {
    v[n-i-1] = s
  }
  for i := 0; i < len(obj.index); i++ {
    if obj.index[i] != vectorSparseIndexMax {
      obj.index[i] = n-obj.index[i]-1
    }
  }
  obj.indexReverse()
  obj.values = v
}
func (obj *SparseRealVector) Slice(i, j int) Vector {
  r := nilSparseRealVector(j-i)
  for i_k := obj.indexFind(i); obj.index[i_k] < j; i_k++ {
    k := obj.index[i_k]
    r.values[k-i] = obj.values[k]
    r.indexInsert(k-i)
  }
  return r
}
func (obj *SparseRealVector) Append(w *SparseRealVector) *SparseRealVector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for entry := range w.RANGE() {
    r.values[obj.n+entry.Index] = entry.Value
    r.indexInsert(obj.n+entry.Index)
  }
  return r
}
func (obj *SparseRealVector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case *Real:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertType(RealType).(*Real)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseRealVector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseRealVector:
    return obj.Append(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for entry := range w.Range() {
      r.values[obj.n+entry.Index] = entry.Value.ConvertType(RealType).(*Real)
      r.indexInsert(obj.n+entry.Index)
    }
    return r
  }
}
func (obj *SparseRealVector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
  obj.indexSwap(i,j)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseRealVector) MapSet(f func(Scalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseRealVector) ElementType() ScalarType {
  return RealType
}
func (obj *SparseRealVector) Variables(order int) error {
  for i, v := range obj.values {
    if err := v.SetVariable(i, obj.n, order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) Permute(pi []int) error {
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
  obj.indexCopy(pi)
  return nil
}
/* sorting
 * -------------------------------------------------------------------------- */
type sortSparseRealVectorByValue struct {
  Value []*Real
}
func (obj sortSparseRealVectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseRealVectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseRealVectorByValue) Less(i, j int) bool {
  return obj.Value[i].GetValue() < obj.Value[j].GetValue()
}
func (obj *SparseRealVector) Sort(reverse bool) {
  r := sortSparseRealVectorByValue{}
  for entry := range obj.RANGE() {
    r.Value = append(r.Value, entry.Value)
  }
  ip := 0
  in := 0
  if reverse {
    in = obj.n - len(obj.values)
  } else {
    ip = obj.n - len(obj.values)
  }
  obj.values = make(map[int]*Real)
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseRealVectorByValue(r))
  }
  for i := 0; i < len(r.Value); i++ {
    if r.Value[i].GetValue() > 0.0 {
      // copy negative values
      obj.values[i+ip] = r.Value[i]
      obj.index [i] = i+ip
    } else {
      // copy negative values
      obj.values[i+in] = r.Value[i]
      obj.index [i] = i+in
    }
  }
  obj.indexSorted = false
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (v SparseRealVector) AsMatrix(n, m int) Matrix {
  return v.ToDenseRealMatrix(n, m)
}
func (obj *SparseRealVector) ToDenseRealMatrix(n, m int) *DenseRealMatrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullDenseRealVector(obj.n)
  for i := 0; i < obj.n; i++ {
    if s, ok := obj.values[i]; ok {
      v[i] = s
    } else {
      v[i] = NullReal()
    }
  }
  matrix := DenseRealMatrix{}
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
func (obj *SparseRealVector) String() string {
  var buffer bytes.Buffer
  buffer.WriteString(fmt.Sprintf("%d:[", obj.n))
  first := true
  for entry := range obj.Range() {
    if !first {
      buffer.WriteString(", ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", entry.Index, entry.Value))
  }
  buffer.WriteString("]")
  return buffer.String()
}
func (obj *SparseRealVector) Table() string {
  var buffer bytes.Buffer
  first := true
  for entry := range obj.Range() {
    if !first {
      buffer.WriteString(" ")
    } else {
      first = false
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", entry.Index, entry.Value))
  }
  if _, ok := obj.values[obj.n-1]; !ok {
    i := obj.n-1
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(fmt.Sprintf("%d:%s", i, ConstReal(0.0)))
  }
  return buffer.String()
}
func (obj *SparseRealVector) Export(filename string) error {
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
func (obj *SparseRealVector) Import(filename string) error {
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
  values := []float64{}
  indices := []int{}
  n := 0
  for scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    if len(fields) == 0 {
      continue
    }
    if len(obj.values) != 0 {
      return fmt.Errorf("invalid sparse table")
    }
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
        values = append(values, v)
      }
    }
  }
  *obj = *NewSparseRealVector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseRealVector) MarshalJSON() ([]byte, error) {
  k := []int{}
  v := []float64{}
  r := struct{
    Index []int
    Value []float64
    Length int}{}
  for entry := range obj.Range() {
    k = append(k, entry.Index)
    v = append(v, entry.Value.GetValue())
  }
  r.Index = k
  r.Value = v
  r.Length = obj.n
  return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseRealVector) UnmarshalJSON(data []byte) error {
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
  *obj = *NewSparseRealVector(r.Index, r.Value, r.Length)
  return nil
}
/* -------------------------------------------------------------------------- */
func (obj *SparseRealVector) nullScalar(s *Real) bool {
  if s.GetValue() != 0.0 {
    return false
  }
  if s.GetOrder() >= 1 {
    for i := 0; i < s.GetN(); i++ {
      if v := s.GetDerivative(i); v != 0.0 {
        return false
      }
    }
  }
  if s.GetOrder() >= 2 {
    for i := 0; i < s.GetN(); i++ {
      for j := 0; j < s.GetN(); j++ {
        if v := s.GetHessian(i, j); v != 0.0 {
          return false
        }
      }
    }
  }
  return true
}
/* iterator
 * -------------------------------------------------------------------------- */
type SparseRealVectorIterator struct {
  *SparseRealVector
  s *Real
  i int
}
func (obj *SparseRealVectorIterator) Get() Scalar {
  return obj.s
}
func (obj *SparseRealVectorIterator) GetConst() ConstScalar {
  return obj.s
}
func (obj *SparseRealVectorIterator) GET() *Real {
  return obj.s
}
func (obj *SparseRealVectorIterator) Ok() bool {
  return obj.i < len(obj.index)
}
func (obj *SparseRealVectorIterator) Next() {
  for obj.i++; obj.Ok() ; obj.i++ {
    i := obj.Index()
    if s := obj.values[i]; obj.nullScalar(s) {
      obj.indexRevoke(obj.i)
      delete(obj.values, i)
    } else {
      obj.s = s; break
    }
  }
}
func (obj *SparseRealVectorIterator) Index() int {
  return obj.index[obj.i]
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseRealVectorJointIterator struct {
  it1 *SparseRealVectorIterator
  it2 VectorConstIterator
  idx int
  s1 *Real
  s2 ConstScalar
}
func (obj *SparseRealVectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseRealVectorJointIterator) Ok() bool {
  return obj.it1.Ok() || obj.it2.Ok()
}
func (obj *SparseRealVectorJointIterator) Next() {
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
    obj.s2 = ConstReal(0.0)
  }
}
func (obj *SparseRealVectorJointIterator) Get() (Scalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseRealVectorJointIterator) GET() (*Real, ConstScalar) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseRealVectorJoint3Iterator struct {
  it1 *SparseRealVectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 *Real
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseRealVectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseRealVectorJoint3Iterator) Ok() bool {
  return obj.it1.Ok() || obj.it2.Ok() || obj.it3.Ok()
}
func (obj *SparseRealVectorJoint3Iterator) Next() {
  ok1 := obj.it1.Ok()
  ok2 := obj.it2.Ok()
  ok3 := obj.it3.Ok()
  obj.s1 = nil
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
      obj.s1 = nil
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
      obj.s1 = nil
      obj.s2 = nil
      obj.s3 = obj.it3.GetConst()
    case obj.idx == i:
      obj.s3 = obj.it3.GetConst()
    }
  }
  if obj.s1 != nil {
    obj.it1.Next()
  }
  if obj.s2 != nil {
    obj.it2.Next()
  } else {
    obj.s2 = ConstReal(0.0)
  }
  if obj.s3 != nil {
    obj.it3.Next()
  } else {
    obj.s3 = ConstReal(0.0)
  }
}
func (obj *SparseRealVectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
func (obj *SparseRealVectorJoint3Iterator) GET() (*Real, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
