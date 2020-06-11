/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2017 Philipp Benner
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
type DenseRealVector []*Real

/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewDenseRealVector(values []float64) DenseRealVector {
	v := nilDenseRealVector(len(values))
	for i, _ := range values {
		v[i] = NewReal(values[i])
	}
	return v
}

// Allocate a new vector. All scalars are set to zero.
func NullDenseRealVector(length int) DenseRealVector {
	v := nilDenseRealVector(length)
	if length > 0 {
		for i := 0; i < length; i++ {
			v[i] = NewReal(0.0)
		}
	}
	return v
}

// Create a empty vector without allocating memory for the scalar variables.
func nilDenseRealVector(length int) DenseRealVector {
	return make(DenseRealVector, length)
}

// Convert vector type.
func AsDenseRealVector(v ConstVector) DenseRealVector {
	switch v_ := v.(type) {
	case DenseRealVector:
		return v_.Clone()
	}
	r := NullDenseRealVector(v.Dim())
	for i := 0; i < v.Dim(); i++ {
		r.AT(i).Set(v.ConstAt(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseRealVector) Clone() DenseRealVector {
	result := make(DenseRealVector, len(v))
	for i, _ := range v {
		result[i] = v[i].Clone()
	}
	return result
}
func (v DenseRealVector) CloneVector() Vector {
	return v.Clone()
}

// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseRealVector) Set(w ConstVector) {
	if v.Dim() != w.Dim() {
		panic("Set(): Vector dimensions do not match!")
	}
	for i := 0; i < w.Dim(); i++ {
		v[i].Set(w.ConstAt(i))
	}
}
func (v DenseRealVector) SET(w DenseRealVector) {
	if v.IDEM(w) {
		return
	}
	if v.Dim() != w.Dim() {
		panic("Set(): Vector dimensions do not match!")
	}
	for i := 0; i < w.Dim(); i++ {
		v[i].SET(w.AT(i))
	}
}
func (v DenseRealVector) IDEM(w DenseRealVector) bool {
	if len(v) != len(w) {
		return false
	}
	if len(v) == 0 {
		return false
	}
	return &v[0] == &w[0]
}

/* const vector methods
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) ValueAt(i int) float64 {
	return v[i].GetValue()
}
func (v DenseRealVector) ConstAt(i int) ConstScalar {
	return v[i]
}
func (v DenseRealVector) ConstSlice(i, j int) ConstVector {
	return v[i:j]
}
func (v DenseRealVector) GetValues() []float64 {
	s := make([]float64, v.Dim())
	for i := 0; i < v.Dim(); i++ {
		s[i] = v.ConstAt(i).GetValue()
	}
	return s
}

/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj DenseRealVector) ConstIterator() VectorConstIterator {
	return obj.ITERATOR()
}
func (obj DenseRealVector) Iterator() VectorIterator {
	return obj.ITERATOR()
}
func (obj DenseRealVector) JointIterator(b ConstVector) VectorJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj DenseRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj DenseRealVector) ITERATOR() *DenseRealVectorIterator {
	r := DenseRealVectorIterator{obj, -1}
	r.Next()
	return &r
}
func (obj DenseRealVector) JOINT_ITERATOR(b ConstVector) *DenseRealVectorJointIterator {
	r := DenseRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
	r.Next()
	return &r
}
func (obj DenseRealVector) JOINT_ITERATOR_(b DenseRealVector) *DenseRealVectorJointIterator_ {
	r := DenseRealVectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, nil, nil}
	r.Next()
	return &r
}

/* -------------------------------------------------------------------------- */
func (v DenseRealVector) Dim() int {
	return len(v)
}
func (v DenseRealVector) At(i int) Scalar {
	return v.AT(i)
}
func (v DenseRealVector) AT(i int) *Real {
	return v[i]
}
func (v DenseRealVector) Reset() {
	for i := 0; i < len(v); i++ {
		v[i].Reset()
	}
}
func (v DenseRealVector) ResetDerivatives() {
	for i := 0; i < len(v); i++ {
		v[i].ResetDerivatives()
	}
}
func (v DenseRealVector) ReverseOrder() {
	n := len(v)
	for i := 0; i < n/2; i++ {
		v[i], v[n-1-i] = v[n-1-i], v[i]
	}
}
func (v DenseRealVector) Slice(i, j int) Vector {
	return v[i:j]
}
func (v DenseRealVector) Append(w DenseRealVector) DenseRealVector {
	return append(v, w...)
}
func (v DenseRealVector) AppendScalar(scalars ...Scalar) Vector {
	for _, scalar := range scalars {
		switch s := scalar.(type) {
		case *Real:
			v = append(v, s)
		default:
			v = append(v, s.ConvertType(RealType).(*Real))
		}
	}
	return v
}
func (v DenseRealVector) AppendVector(w_ Vector) Vector {
	switch w := w_.(type) {
	case DenseRealVector:
		return append(v, w...)
	default:
		for i := 0; i < w.Dim(); i++ {
			v = append(v, w.At(i).ConvertType(RealType).(*Real))
		}
		return v
	}
}
func (v DenseRealVector) Swap(i, j int) {
	v[i], v[j] = v[j], v[i]
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) Map(f func(Scalar)) {
	for i := 0; i < len(v); i++ {
		f(v[i])
	}
}
func (v DenseRealVector) MapSet(f func(ConstScalar) Scalar) {
	for i := 0; i < len(v); i++ {
		v[i].Set(f(v.ConstAt(i)))
	}
}
func (v DenseRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	for i := 0; i < len(v); i++ {
		r = f(r, v.ConstAt(i))
	}
	return r
}
func (v DenseRealVector) ElementType() ScalarType {
	return RealType
}
func (v DenseRealVector) Variables(order int) error {
	for i, _ := range v {
		if err := v[i].SetVariable(i, len(v), order); err != nil {
			return err
		}
	}
	return nil
}

/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) Permute(pi []int) error {
	if len(pi) != len(v) {
		return errors.New("Permute(): permutation vector has invalid length!")
	}
	// permute vector
	for i := 0; i < len(v); i++ {
		if pi[i] < 0 || pi[i] >= len(v) {
			return errors.New("Permute(): invalid permutation")
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
type sortDenseRealVectorByValue DenseRealVector

func (v sortDenseRealVectorByValue) Len() int           { return len(v) }
func (v sortDenseRealVectorByValue) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v sortDenseRealVectorByValue) Less(i, j int) bool { return v[i].GetValue() < v[j].GetValue() }
func (v DenseRealVector) Sort(reverse bool) {
	if reverse {
		sort.Sort(sort.Reverse(sortDenseRealVectorByValue(v)))
	} else {
		sort.Sort(sortDenseRealVectorByValue(v))
	}
}

/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseRealVector) AsMatrix(n, m int) Matrix {
	return v.ToDenseRealMatrix(n, m)
}
func (v DenseRealVector) ToDenseRealMatrix(n, m int) *DenseRealMatrix {
	if n*m != len(v) {
		panic("Matrix dimension does not fit input vector!")
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
func (v DenseRealVector) String() string {
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
func (v DenseRealVector) Table() string {
	var buffer bytes.Buffer
	for i, _ := range v {
		if i != 0 {
			buffer.WriteString(" ")
		}
		buffer.WriteString(v[i].String())
	}
	return buffer.String()
}
func (v DenseRealVector) Export(filename string) error {
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
func (v *DenseRealVector) Import(filename string) error {
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
	*v = DenseRealVector{}
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
			*v = append(*v, NewReal(value))
		}
	}
	return nil
}

/* json
 * -------------------------------------------------------------------------- */
func (obj DenseRealVector) MarshalJSON() ([]byte, error) {
	r := []*Real{}
	r = obj
	return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseRealVector) UnmarshalJSON(data []byte) error {
	r := []*Real{}
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	*obj = nilDenseRealVector(len(r))
	for i := 0; i < len(r); i++ {
		(*obj)[i] = r[i]
	}
	return nil
}

/* iterator
 * -------------------------------------------------------------------------- */
type DenseRealVectorIterator struct {
	v DenseRealVector
	i int
}

func (obj *DenseRealVectorIterator) Get() Scalar {
	return obj.GET()
}
func (obj *DenseRealVectorIterator) GetConst() ConstScalar {
	return obj.GET()
}
func (obj *DenseRealVectorIterator) GetValue() float64 {
	return obj.GET().GetValue()
}
func (obj *DenseRealVectorIterator) GET() *Real {
	return obj.v[obj.i]
}
func (obj *DenseRealVectorIterator) Ok() bool {
	return obj.i < len(obj.v)
}
func (obj *DenseRealVectorIterator) Next() {
	obj.i++
}
func (obj *DenseRealVectorIterator) Index() int {
	return obj.i
}
func (obj *DenseRealVectorIterator) Clone() *DenseRealVectorIterator {
	return &DenseRealVectorIterator{obj.v, obj.i}
}
func (obj *DenseRealVectorIterator) CloneConstIterator() VectorConstIterator {
	return &DenseRealVectorIterator{obj.v, obj.i}
}
func (obj *DenseRealVectorIterator) CloneIterator() VectorIterator {
	return &DenseRealVectorIterator{obj.v, obj.i}
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseRealVectorJointIterator struct {
	it1 *DenseRealVectorIterator
	it2 VectorConstIterator
	idx int
	s1  *Real
	s2  ConstScalar
}

func (obj *DenseRealVectorJointIterator) Index() int {
	return obj.idx
}
func (obj *DenseRealVectorJointIterator) Ok() bool {
	return !(obj.s1 == nil || obj.s1.GetValue() == 0.0) ||
		!(obj.s2 == nil || obj.s2.GetValue() == 0.0)
}
func (obj *DenseRealVectorJointIterator) Next() {
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
func (obj *DenseRealVectorJointIterator) Get() (Scalar, ConstScalar) {
	return obj.GET()
}
func (obj *DenseRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
	return obj.GET()
}
func (obj *DenseRealVectorJointIterator) GetValue() (float64, float64) {
	a, b := obj.GET()
	return a.GetValue(), b.GetValue()
}
func (obj *DenseRealVectorJointIterator) GET() (*Real, ConstScalar) {
	if obj.s1 == nil {
		return nil, obj.s2
	} else {
		return obj.s1, obj.s2
	}
}
func (obj *DenseRealVectorJointIterator) Clone() *DenseRealVectorJointIterator {
	r := DenseRealVectorJointIterator{}
	r.it1 = obj.it1.Clone()
	r.it2 = obj.it2.CloneConstIterator()
	r.idx = obj.idx
	r.s1 = obj.s1
	r.s2 = obj.s2
	return &r
}
func (obj *DenseRealVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
	return obj.Clone()
}
func (obj *DenseRealVectorJointIterator) CloneJointIterator() VectorJointIterator {
	return obj.Clone()
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseRealVectorJointIterator_ struct {
	it1 *DenseRealVectorIterator
	it2 *DenseRealVectorIterator
	idx int
	s1  *Real
	s2  *Real
}

func (obj *DenseRealVectorJointIterator_) Index() int {
	return obj.idx
}
func (obj *DenseRealVectorJointIterator_) Ok() bool {
	return obj.s1 != nil || obj.s2 != nil
}
func (obj *DenseRealVectorJointIterator_) Next() {
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
func (obj *DenseRealVectorJointIterator_) GET() (*Real, *Real) {
	return obj.s1, obj.s2
}
