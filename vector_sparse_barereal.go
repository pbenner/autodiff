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
type SparseBareRealVector struct {
  vectorSparseIndexSlice
  values map[int]*BareReal
  n int
}
/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseBareRealVector(indices []int, values []float64, n int) *SparseBareRealVector {
  r := nilSparseBareRealVector(n)
  for i, k := range indices {
    if k >= n {
      panic("index larger than vector dimension")
    }
    if values[i] != 0.0 {
      r.values[k] = NewBareReal(values[i])
      r.indexInsert(k)
    }
  }
  return r
}
// Allocate a new vector. All scalars are set to zero.
func NullSparseBareRealVector(length int) *SparseBareRealVector {
  v := nilSparseBareRealVector(length)
  return v
}
// Create a empty vector without allocating memory for the scalar variables.
func nilSparseBareRealVector(length int) *SparseBareRealVector {
  return &SparseBareRealVector{values: make(map[int]*BareReal), n: length}
}
// Convert vector type.
func AsSparseBareRealVector(v ConstVector) *SparseBareRealVector {
  switch v_ := v.(type) {
  case *SparseBareRealVector:
    return v_
  }
  r := NullSparseBareRealVector(v.Dim())
  for entry := range v.ConstRange() {
    r.AT(entry.Index).Set(entry.Value)
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (obj *SparseBareRealVector) Clone() *SparseBareRealVector {
  r := nilSparseBareRealVector(obj.n)
  for i, v := range obj.values {
    r.values[i] = v.Clone()
  }
  r.vectorSparseIndexSlice = obj.indexClone()
  return r
}
func (obj *SparseBareRealVector) CloneVector() Vector {
  return obj.Clone()
}
// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseBareRealVector) Set(x ConstVector) {
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
func (obj *SparseBareRealVector) ValueAt(i int) float64 {
  if v, ok := obj.values[i]; ok {
    return v.GetValue()
  } else {
    return 0.0
  }
}
func (obj *SparseBareRealVector) ConstAt(i int) ConstScalar {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    return ConstReal(0.0)
  }
}
func (obj *SparseBareRealVector) ConstSlice(i, j int) ConstVector {
  return obj.Slice(i, j)
}
func (obj *SparseBareRealVector) GetValues() []float64 {
  r := make([]float64, obj.Dim())
  for i, v := range obj.values {
    r[i] = v.GetValue()
  }
  return r
}
func (obj *SparseBareRealVector) ConstRange() chan VectorConstRangeType {
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
func (obj *SparseBareRealVector) ConstIterator() VectorConstIterator {
  r := SparseBareRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) JointIterator(b ConstVector) VectorJointIterator {
  r := SparseBareRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
  r := SparseBareRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) Iterator() VectorIterator {
  r := SparseBareRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) ITERATOR() *SparseBareRealVectorIterator {
  r := SparseBareRealVectorIterator{obj, nil, -1}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) CONST_JOINT_ITERATOR(b ConstVector) *SparseBareRealVectorJointIterator {
  r := SparseBareRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
  r.Next()
  return &r
}
func (obj *SparseBareRealVector) CONST_JOINT3_ITERATOR(b, c ConstVector) *SparseBareRealVectorJoint3Iterator {
  r := SparseBareRealVectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, nil, nil, nil}
  r.Next()
  return &r
}
/* range methods
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) Range() chan VectorRangeType {
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
func (obj *SparseBareRealVector) JointRange(b ConstVector) chan VectorJointRangeType {
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
type SparseBareRealVectorRange struct {
  Index int
  Value *BareReal
}
func (obj *SparseBareRealVector) RANGE() chan SparseBareRealVectorRange {
  channel := make(chan SparseBareRealVectorRange)
  go func() {
    obj.indexSort()
    for k, i := range obj.index {
      if s := obj.values[i]; obj.nullScalar(s) {
        obj.indexRevoke(k)
        delete(obj.values, i)
      } else {
        channel <- SparseBareRealVectorRange{i, s}
      }
    }
    close(channel)
  }()
  return channel
}
type SparseBareRealVectorJointRange struct {
  Index int
  Value1 *BareReal
  Value2 ConstScalar
}
func (obj *SparseBareRealVector) JOINT_RANGE(b ConstVector) chan SparseBareRealVectorJointRange {
  channel := make(chan SparseBareRealVectorJointRange)
  go func() {
    c1 := obj. RANGE()
    c2 := b.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    for ok1 || ok2 {
      r := SparseBareRealVectorJointRange{}
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
type SparseBareRealVectorJointRange3 struct {
  Index int
  Value1 *BareReal
  Value2 ConstScalar
  Value3 ConstScalar
}
func (obj *SparseBareRealVector) JOINT_RANGE3(b, c ConstVector) chan SparseBareRealVectorJointRange3 {
  channel := make(chan SparseBareRealVectorJointRange3)
  go func() {
    c1 := obj. RANGE()
    c2 := b.ConstRange()
    c3 := c.ConstRange()
    r1, ok1 := <- c1
    r2, ok2 := <- c2
    r3, ok3 := <- c3
    for ok1 || ok2 || ok3 {
      r := SparseBareRealVectorJointRange3{}
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
func (obj *SparseBareRealVector) Dim() int {
  return obj.n
}
func (obj *SparseBareRealVector) At(i int) Scalar {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullBareReal()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseBareRealVector) AT(i int) *BareReal {
  if v, ok := obj.values[i]; ok {
    return v
  } else {
    v = NullBareReal()
    obj.values[i] = v
    obj.indexInsert(i)
    return v
  }
}
func (obj *SparseBareRealVector) Reset() {
  for _, v := range obj.values {
    v.Reset()
  }
}
func (obj *SparseBareRealVector) ResetDerivatives() {
  for _, v := range obj.values {
    v.ResetDerivatives()
  }
}
func (obj *SparseBareRealVector) ReverseOrder() {
  n := obj.Dim()
  v := make(map[int]*BareReal)
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
func (obj *SparseBareRealVector) Slice(i, j int) Vector {
  r := nilSparseBareRealVector(j-i)
  for i_k := obj.indexFind(i); obj.index[i_k] < j; i_k++ {
    k := obj.index[i_k]
    r.values[k-i] = obj.values[k]
    r.indexInsert(k-i)
  }
  return r
}
func (obj *SparseBareRealVector) Append(w *SparseBareRealVector) *SparseBareRealVector {
  r := obj.Clone()
  r.n = obj.n + w.Dim()
  for entry := range w.RANGE() {
    r.values[obj.n+entry.Index] = entry.Value
    r.indexInsert(obj.n+entry.Index)
  }
  return r
}
func (obj *SparseBareRealVector) AppendScalar(scalars ...Scalar) Vector {
  r := obj.Clone()
  r.n = obj.n + len(scalars)
  for i, scalar := range scalars {
    switch s := scalar.(type) {
    case *BareReal:
      r.values[obj.n+i] = s
    default:
      r.values[obj.n+i] = s.ConvertType(BareRealType).(*BareReal)
    }
    r.indexInsert(obj.n+i)
  }
  return r
}
func (obj *SparseBareRealVector) AppendVector(w_ Vector) Vector {
  switch w := w_.(type) {
  case *SparseBareRealVector:
    return obj.Append(w)
  default:
    r := obj.Clone()
    r.n = obj.n + w.Dim()
    for entry := range w.Range() {
      r.values[obj.n+entry.Index] = entry.Value.ConvertType(BareRealType).(*BareReal)
      r.indexInsert(obj.n+entry.Index)
    }
    return r
  }
}
func (obj *SparseBareRealVector) Swap(i, j int) {
  obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
  obj.indexSwap(i,j)
}
/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) Map(f func(Scalar)) {
  for _, v := range obj.values {
    f(v)
  }
}
func (obj *SparseBareRealVector) MapSet(f func(Scalar) Scalar) {
  for _, v := range obj.values {
    v.Set(f(v))
  }
}
func (obj *SparseBareRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for _, v := range obj.values {
    r = f(r, v)
  }
  return r
}
func (obj *SparseBareRealVector) ElementType() ScalarType {
  return BareRealType
}
func (obj *SparseBareRealVector) Variables(order int) error {
  for i, v := range obj.values {
    if err := v.SetVariable(i, obj.n, order); err != nil {
      return err
    }
  }
  return nil
}
/* permutations
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) Permute(pi []int) error {
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
type sortSparseBareRealVectorByValue struct {
  Value []*BareReal
}
func (obj sortSparseBareRealVectorByValue) Len() int {
  return len(obj.Value)
}
func (obj sortSparseBareRealVectorByValue) Swap(i, j int) {
  obj.Value[i], obj.Value[j] = obj.Value[j], obj.Value[i]
}
func (obj sortSparseBareRealVectorByValue) Less(i, j int) bool {
  return obj.Value[i].GetValue() < obj.Value[j].GetValue()
}
func (obj *SparseBareRealVector) Sort(reverse bool) {
  r := sortSparseBareRealVectorByValue{}
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
  obj.values = make(map[int]*BareReal)
  if reverse {
    sort.Sort(sort.Reverse(r))
  } else {
    sort.Sort(sortSparseBareRealVectorByValue(r))
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
func (v SparseBareRealVector) AsMatrix(n, m int) Matrix {
  return v.ToDenseBareRealMatrix(n, m)
}
func (obj *SparseBareRealVector) ToDenseBareRealMatrix(n, m int) *DenseBareRealMatrix {
  if n*m != obj.n {
    panic("Matrix dimension does not fit input vector!")
  }
  v := NullDenseBareRealVector(obj.n)
  for i := 0; i < obj.n; i++ {
    if s, ok := obj.values[i]; ok {
      v[i] = s
    } else {
      v[i] = NullBareReal()
    }
  }
  matrix := DenseBareRealMatrix{}
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
func (obj *SparseBareRealVector) String() string {
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
func (obj *SparseBareRealVector) Table() string {
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
func (obj *SparseBareRealVector) Export(filename string) error {
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
func (obj *SparseBareRealVector) Import(filename string) error {
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
  *obj = *NewSparseBareRealVector(indices, values, n)
  return nil
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) MarshalJSON() ([]byte, error) {
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
func (obj *SparseBareRealVector) UnmarshalJSON(data []byte) error {
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
  *obj = *NewSparseBareRealVector(r.Index, r.Value, r.Length)
  return nil
}
/* -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) nullScalar(s *BareReal) bool {
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
type SparseBareRealVectorIterator struct {
  *SparseBareRealVector
  s *BareReal
  i int
}
func (obj *SparseBareRealVectorIterator) Get() Scalar {
  return obj.s
}
func (obj *SparseBareRealVectorIterator) GetConst() ConstScalar {
  return obj.s
}
func (obj *SparseBareRealVectorIterator) GET() *BareReal {
  return obj.s
}
func (obj *SparseBareRealVectorIterator) Ok() bool {
  return obj.i < len(obj.index)
}
func (obj *SparseBareRealVectorIterator) Next() {
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
func (obj *SparseBareRealVectorIterator) Index() int {
  return obj.index[obj.i]
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJointIterator struct {
  it1 *SparseBareRealVectorIterator
  it2 VectorConstIterator
  idx int
  s1 *BareReal
  s2 ConstScalar
}
func (obj *SparseBareRealVectorJointIterator) Index() int {
  return obj.idx
}
func (obj *SparseBareRealVectorJointIterator) Ok() bool {
  return obj.it1.Ok() || obj.it2.Ok()
}
func (obj *SparseBareRealVectorJointIterator) Next() {
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
func (obj *SparseBareRealVectorJointIterator) Get() (Scalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseBareRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
  return obj.s1, obj.s2
}
func (obj *SparseBareRealVectorJointIterator) GET() (*BareReal, ConstScalar) {
  return obj.s1, obj.s2
}
/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJoint3Iterator struct {
  it1 *SparseBareRealVectorIterator
  it2 VectorConstIterator
  it3 VectorConstIterator
  idx int
  s1 *BareReal
  s2 ConstScalar
  s3 ConstScalar
}
func (obj *SparseBareRealVectorJoint3Iterator) Index() int {
  return obj.idx
}
func (obj *SparseBareRealVectorJoint3Iterator) Ok() bool {
  return obj.it1.Ok() || obj.it2.Ok() || obj.it3.Ok()
}
func (obj *SparseBareRealVectorJoint3Iterator) Next() {
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
func (obj *SparseBareRealVectorJoint3Iterator) Get() (Scalar, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
func (obj *SparseBareRealVectorJoint3Iterator) GET() (*BareReal, ConstScalar, ConstScalar) {
  return obj.s1, obj.s2, obj.s3
}
