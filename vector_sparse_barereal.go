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
/* vector type declaration
 * -------------------------------------------------------------------------- */
type SparseBareRealVector struct {
	vectorSparseIndex
	values map[int]*BareReal
	n      int
}

/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewSparseBareRealVector(indices []int, values []float64, n int) *SparseBareRealVector {
	if len(indices) != len(values) {
		panic("number of indices does not match number of values")
	}
	r := nilSparseBareRealVector(n)
	for i, k := range indices {
		if k >= n {
			panic("index larger than vector dimension")
		}
		if _, ok := r.values[k]; ok {
			panic("index appeared multiple times")
		} else {
			if values[i] != 0.0 {
				r.values[k] = NewBareReal(values[i])
				r.indexInsert(k)
			}
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
		return v_.Clone()
	}
	r := NullSparseBareRealVector(v.Dim())
	for it := v.ConstIterator(); it.Ok(); it.Next() {
		r.AT(it.Index()).Set(it.GetConst())
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
	r.vectorSparseIndex = obj.indexClone()
	return r
}
func (obj *SparseBareRealVector) CloneVector() Vector {
	return obj.Clone()
}

// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (obj *SparseBareRealVector) Set(x ConstVector) {
	if obj == x {
		return
	}
	if obj.Dim() != x.Dim() {
		panic("Set(): Vector dimensions do not match!")
	}
	for it := obj.JOINT_ITERATOR(x); it.Ok(); it.Next() {
		s1, s2 := it.Get()
		switch {
		case s1 != nil && s2 != nil:
			s1.Set(s2)
		case s1 != nil:
			s1.SetValue(0.0)
		default:
			obj.AT(it.Index()).Set(s2)
		}
	}
}
func (obj *SparseBareRealVector) SET(x *SparseBareRealVector) {
	if obj == x {
		return
	}
	if obj.Dim() != x.Dim() {
		panic("Set(): Vector dimensions do not match!")
	}
	for it := obj.JOINT_ITERATOR_(x); it.Ok(); it.Next() {
		s1, s2 := it.GET()
		switch {
		case s1 != nil && s2 != nil:
			s1.SET(s2)
		case s1 != nil:
			s1.SetValue(0.0)
		default:
			obj.AT(it.Index()).SET(s2)
		}
	}
}
func (obj *SparseBareRealVector) IDEM(x *SparseBareRealVector) bool {
	return obj == x
}

/* const vector methods
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) ValueAt(i int) float64 {
	if i < 0 || i >= obj.Dim() {
		panic("index out of bounds")
	}
	if v, ok := obj.values[i]; ok {
		return v.GetValue()
	} else {
		return 0.0
	}
}
func (obj *SparseBareRealVector) ConstAt(i int) ConstScalar {
	if i < 0 || i >= obj.Dim() {
		panic("index out of bounds")
	}
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

/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) ConstIterator() VectorConstIterator {
	return obj.ITERATOR()
}
func (obj *SparseBareRealVector) Iterator() VectorIterator {
	return obj.ITERATOR()
}
func (obj *SparseBareRealVector) JointIterator(b ConstVector) VectorJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj *SparseBareRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj *SparseBareRealVector) ITERATOR() *SparseBareRealVectorIterator {
	r := SparseBareRealVectorIterator{obj.indexIterator(), obj}
	return &r
}
func (obj *SparseBareRealVector) JOINT_ITERATOR(b ConstVector) *SparseBareRealVectorJointIterator {
	r := SparseBareRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
	r.Next()
	return &r
}
func (obj *SparseBareRealVector) JOINT3_ITERATOR(b, c ConstVector) *SparseBareRealVectorJoint3Iterator {
	r := SparseBareRealVectorJoint3Iterator{obj.ITERATOR(), b.ConstIterator(), c.ConstIterator(), -1, nil, nil, nil}
	r.Next()
	return &r
}
func (obj *SparseBareRealVector) JOINT_ITERATOR_(b *SparseBareRealVector) *SparseBareRealVectorJointIterator_ {
	r := SparseBareRealVectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, nil, nil}
	r.Next()
	return &r
}
func (obj *SparseBareRealVector) JOINT3_ITERATOR_(b, c *SparseBareRealVector) *SparseBareRealVectorJoint3Iterator_ {
	r := SparseBareRealVectorJoint3Iterator_{obj.ITERATOR(), b.ITERATOR(), c.ITERATOR(), -1, nil, nil, nil}
	r.Next()
	return &r
}

/* -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) Dim() int {
	return obj.n
}
func (obj *SparseBareRealVector) At(i int) Scalar {
	return obj.AT(i)
}
func (obj *SparseBareRealVector) AT(i int) *BareReal {
	if i < 0 || i >= obj.Dim() {
		panic("index out of bounds")
	}
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
	values := make(map[int]*BareReal)
	index := vectorSparseIndex{}
	for i, s := range obj.values {
		j := n - i - 1
		values[j] = s
		index.indexInsert(j)
	}
	obj.values = values
	obj.vectorSparseIndex = index
}
func (obj *SparseBareRealVector) Slice(i, j int) Vector {
	r := nilSparseBareRealVector(j - i)
	for it := obj.indexIteratorFrom(i); it.Ok(); it.Next() {
		if it.Get() >= j {
			break
		}
		k := it.Get()
		r.values[k-i] = obj.values[k]
		r.indexInsert(k - i)
	}
	return r
}
func (obj *SparseBareRealVector) Append(w *SparseBareRealVector) *SparseBareRealVector {
	r := obj.Clone()
	r.n = obj.n + w.Dim()
	for it := w.ITERATOR(); it.Ok(); it.Next() {
		i := obj.n + it.Index()
		r.values[i] = it.GET()
		r.indexInsert(i)
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
		r.indexInsert(obj.n + i)
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
		for it := w.Iterator(); it.Ok(); it.Next() {
			r.values[obj.n+it.Index()] = it.Get().ConvertType(BareRealType).(*BareReal)
			r.indexInsert(obj.n + it.Index())
		}
		return r
	}
}
func (obj *SparseBareRealVector) Swap(i, j int) {
	obj.values[i], obj.values[j] = obj.values[j], obj.values[i]
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (obj *SparseBareRealVector) Map(f func(Scalar)) {
	for _, v := range obj.values {
		f(v)
	}
}
func (obj *SparseBareRealVector) MapSet(f func(ConstScalar) Scalar) {
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
			} else if ok1 {
				obj.values[pi[i]] = obj.values[i]
				delete(obj.values, i)
			} else if ok2 {
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
	obj.values = make(map[int]*BareReal)
	obj.vectorSparseIndex = vectorSparseIndex{}
	if reverse {
		sort.Sort(sort.Reverse(r))
	} else {
		sort.Sort(sortSparseBareRealVectorByValue(r))
	}
	for i := 0; i < len(r.Value); i++ {
		if r.Value[i].GetValue() > 0.0 {
			// copy negative values
			obj.values[i+ip] = r.Value[i]
			obj.indexInsert(i + ip)
		} else {
			// copy negative values
			obj.values[i+in] = r.Value[i]
			obj.indexInsert(i + in)
		}
	}
}

/* type conversion
 * -------------------------------------------------------------------------- */
func (v *SparseBareRealVector) AsMatrix(n, m int) Matrix {
	return v.ToDenseBareRealMatrix(n, m)
}
func (obj *SparseBareRealVector) ToDenseBareRealMatrix(n, m int) *DenseBareRealMatrix {
	if n*m != obj.n {
		panic("Matrix dimension does not fit input vector!")
	}
	v := NullDenseBareRealVector(obj.n)
	for it := obj.ITERATOR(); it.Ok(); it.Next() {
		v.At(it.Index()).Set(it.GET())
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
func (obj *SparseBareRealVector) Table() string {
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
		i := obj.n - 1
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
					n = int(k) + 1
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
	r := struct {
		Index  []int
		Value  []float64
		Length int
	}{}
	for it := obj.ConstIterator(); it.Ok(); it.Next() {
		k = append(k, it.Index())
		v = append(v, it.GetValue())
	}
	r.Index = k
	r.Value = v
	r.Length = obj.n
	return json.MarshalIndent(r, "", "  ")
}
func (obj *SparseBareRealVector) UnmarshalJSON(data []byte) error {
	r := struct {
		Index  []int
		Value  []float64
		Length int
	}{}
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
	vectorSparseIndexIterator
	v *SparseBareRealVector
}

func (obj *SparseBareRealVectorIterator) Get() Scalar {
	if v := obj.GET(); v == (*BareReal)(nil) {
		return nil
	} else {
		return v
	}
}
func (obj *SparseBareRealVectorIterator) GetConst() ConstScalar {
	if v, ok := obj.v.values[obj.Index()]; ok {
		return v
	} else {
		return nil
	}
}
func (obj *SparseBareRealVectorIterator) GetValue() float64 {
	if v, ok := obj.v.values[obj.Index()]; ok {
		return v.GetValue()
	} else {
		return 0.0
	}
}
func (obj *SparseBareRealVectorIterator) GET() *BareReal {
	if v, ok := obj.v.values[obj.Index()]; ok {
		return v
	} else {
		return nil
	}
}
func (obj *SparseBareRealVectorIterator) Next() {
	obj.vectorSparseIndexIterator.Next()
	for obj.Ok() && obj.v.nullScalar(obj.GET()) {
		i := obj.Index()
		obj.vectorSparseIndexIterator.Next()
		delete(obj.v.values, i)
		obj.v.indexDelete(i)
	}
}
func (obj *SparseBareRealVectorIterator) Index() int {
	return obj.vectorSparseIndexIterator.Get()
}
func (obj *SparseBareRealVectorIterator) Clone() *SparseBareRealVectorIterator {
	return &SparseBareRealVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseBareRealVectorIterator) CloneConstIterator() VectorConstIterator {
	return &SparseBareRealVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}
func (obj *SparseBareRealVectorIterator) CloneIterator() VectorIterator {
	return &SparseBareRealVectorIterator{*obj.vectorSparseIndexIterator.Clone(), obj.v}
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJointIterator struct {
	it1 *SparseBareRealVectorIterator
	it2 VectorConstIterator
	idx int
	s1  *BareReal
	s2  ConstScalar
}

func (obj *SparseBareRealVectorJointIterator) Index() int {
	return obj.idx
}
func (obj *SparseBareRealVectorJointIterator) Ok() bool {
	return !(obj.s1 == nil || obj.s1.GetValue() == 0.0) ||
		!(obj.s2 == nil || obj.s2.GetValue() == 0.0)
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
	if obj.s1 == nil {
		return nil, obj.s2
	} else {
		return obj.s1, obj.s2
	}
}
func (obj *SparseBareRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
	if obj.s1 == nil {
		return nil, obj.s2
	} else {
		return obj.s1, obj.s2
	}
}
func (obj *SparseBareRealVectorJointIterator) GetValue() (float64, float64) {
	v1 := 0.0
	v2 := 0.0
	if obj.s1 != nil {
		v1 = obj.s1.GetValue()
	}
	if obj.s2 != nil {
		v2 = obj.s2.GetValue()
	}
	return v1, v2
}
func (obj *SparseBareRealVectorJointIterator) GET() (*BareReal, ConstScalar) {
	return obj.s1, obj.s2
}
func (obj *SparseBareRealVectorJointIterator) Clone() *SparseBareRealVectorJointIterator {
	r := SparseBareRealVectorJointIterator{}
	r.it1 = obj.it1.Clone()
	r.it2 = obj.it2.CloneConstIterator()
	r.idx = obj.idx
	r.s1 = obj.s1
	r.s2 = obj.s2
	return &r
}
func (obj *SparseBareRealVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
	return obj.Clone()
}
func (obj *SparseBareRealVectorJointIterator) CloneJointIterator() VectorJointIterator {
	return obj.Clone()
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJoint3Iterator struct {
	it1 *SparseBareRealVectorIterator
	it2 VectorConstIterator
	it3 VectorConstIterator
	idx int
	s1  *BareReal
	s2  ConstScalar
	s3  ConstScalar
}

func (obj *SparseBareRealVectorJoint3Iterator) Index() int {
	return obj.idx
}
func (obj *SparseBareRealVectorJoint3Iterator) Ok() bool {
	return !(obj.s1 == nil || obj.s1.GetValue() == 0.0) ||
		!(obj.s2 == nil || obj.s2.GetValue() == 0.0) ||
		!(obj.s3 == nil || obj.s3.GetValue() == 0.0)
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
	if obj.s1 == nil {
		return nil, obj.s2, obj.s3
	} else {
		return obj.s1, obj.s2, obj.s3
	}
}
func (obj *SparseBareRealVectorJoint3Iterator) GET() (*BareReal, ConstScalar, ConstScalar) {
	return obj.s1, obj.s2, obj.s3
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJointIterator_ struct {
	it1 *SparseBareRealVectorIterator
	it2 *SparseBareRealVectorIterator
	idx int
	s1  *BareReal
	s2  *BareReal
}

func (obj *SparseBareRealVectorJointIterator_) Index() int {
	return obj.idx
}
func (obj *SparseBareRealVectorJointIterator_) Ok() bool {
	return obj.s1 != nil || obj.s2 != nil
}
func (obj *SparseBareRealVectorJointIterator_) Next() {
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
func (obj *SparseBareRealVectorJointIterator_) GET() (*BareReal, *BareReal) {
	return obj.s1, obj.s2
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type SparseBareRealVectorJoint3Iterator_ struct {
	it1 *SparseBareRealVectorIterator
	it2 *SparseBareRealVectorIterator
	it3 *SparseBareRealVectorIterator
	idx int
	s1  *BareReal
	s2  *BareReal
	s3  *BareReal
}

func (obj *SparseBareRealVectorJoint3Iterator_) Index() int {
	return obj.idx
}
func (obj *SparseBareRealVectorJoint3Iterator_) Ok() bool {
	return obj.s1 != nil || obj.s2 != nil || obj.s3 != nil
}
func (obj *SparseBareRealVectorJoint3Iterator_) Next() {
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
			obj.s1 = nil
			obj.s2 = nil
			obj.s3 = obj.it3.GET()
		case obj.idx == i:
			obj.s3 = obj.it3.GET()
		}
	}
	if obj.s1 != nil {
		obj.it1.Next()
	}
	if obj.s2 != nil {
		obj.it2.Next()
	}
	if obj.s3 != nil {
		obj.it3.Next()
	}
}
func (obj *SparseBareRealVectorJoint3Iterator_) GET() (*BareReal, *BareReal, *BareReal) {
	return obj.s1, obj.s2, obj.s3
}
