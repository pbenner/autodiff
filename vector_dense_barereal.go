//#define STORE_PTR 1
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
type DenseBareRealVector []BareReal

/* constructors
 * -------------------------------------------------------------------------- */
// Allocate a new vector. Scalars are set to the given values.
func NewDenseBareRealVector(values []float64) DenseBareRealVector {
	v := nilDenseBareRealVector(len(values))
	for i, _ := range values {
		v[i] = *NewBareReal(values[i])
	}
	return v
}

// Allocate a new vector. All scalars are set to zero.
func NullDenseBareRealVector(length int) DenseBareRealVector {
	v := nilDenseBareRealVector(length)
	if length > 0 {
		for i := 0; i < length; i++ {
			v[i] = *NewBareReal(0.0)
		}
	}
	return v
}

// Create a empty vector without allocating memory for the scalar variables.
func nilDenseBareRealVector(length int) DenseBareRealVector {
	return make(DenseBareRealVector, length)
}

// Convert vector type.
func AsDenseBareRealVector(v ConstVector) DenseBareRealVector {
	switch v_ := v.(type) {
	case DenseBareRealVector:
		return v_.Clone()
	}
	r := NullDenseBareRealVector(v.Dim())
	for i := 0; i < v.Dim(); i++ {
		r.AT(i).Set(v.ConstAt(i))
	}
	return r
}

/* -------------------------------------------------------------------------- */
// Create a deep copy of the vector.
func (v DenseBareRealVector) Clone() DenseBareRealVector {
	result := make(DenseBareRealVector, len(v))
	for i, _ := range v {
		result[i] = *v[i].Clone()
	}
	return result
}
func (v DenseBareRealVector) CloneVector() Vector {
	return v.Clone()
}

// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v DenseBareRealVector) Set(w ConstVector) {
	if v.Dim() != w.Dim() {
		panic("Set(): Vector dimensions do not match!")
	}
	for i := 0; i < w.Dim(); i++ {
		v[i].Set(w.ConstAt(i))
	}
}
func (v DenseBareRealVector) SET(w DenseBareRealVector) {
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
func (v DenseBareRealVector) IDEM(w DenseBareRealVector) bool {
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
func (v DenseBareRealVector) ValueAt(i int) float64 {
	return v[i].GetValue()
}
func (v DenseBareRealVector) ConstAt(i int) ConstScalar {
	return v[i]
}
func (v DenseBareRealVector) ConstSlice(i, j int) ConstVector {
	return v[i:j]
}
func (v DenseBareRealVector) GetValues() []float64 {
	s := make([]float64, v.Dim())
	for i := 0; i < v.Dim(); i++ {
		s[i] = v.ConstAt(i).GetValue()
	}
	return s
}

/* iterator methods
 * -------------------------------------------------------------------------- */
func (obj DenseBareRealVector) ConstIterator() VectorConstIterator {
	return obj.ITERATOR()
}
func (obj DenseBareRealVector) Iterator() VectorIterator {
	return obj.ITERATOR()
}
func (obj DenseBareRealVector) JointIterator(b ConstVector) VectorJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj DenseBareRealVector) ConstJointIterator(b ConstVector) VectorConstJointIterator {
	return obj.JOINT_ITERATOR(b)
}
func (obj DenseBareRealVector) ITERATOR() *DenseBareRealVectorIterator {
	r := DenseBareRealVectorIterator{obj, -1}
	r.Next()
	return &r
}
func (obj DenseBareRealVector) JOINT_ITERATOR(b ConstVector) *DenseBareRealVectorJointIterator {
	r := DenseBareRealVectorJointIterator{obj.ITERATOR(), b.ConstIterator(), -1, nil, nil}
	r.Next()
	return &r
}
func (obj DenseBareRealVector) JOINT_ITERATOR_(b DenseBareRealVector) *DenseBareRealVectorJointIterator_ {
	r := DenseBareRealVectorJointIterator_{obj.ITERATOR(), b.ITERATOR(), -1, nil, nil}
	r.Next()
	return &r
}

/* -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Dim() int {
	return len(v)
}
func (v DenseBareRealVector) At(i int) Scalar {
	return v.AT(i)
}
func (v DenseBareRealVector) AT(i int) *BareReal {
	return &v[i]
}
func (v DenseBareRealVector) Reset() {
	for i := 0; i < len(v); i++ {
		v[i].Reset()
	}
}
func (v DenseBareRealVector) ResetDerivatives() {
	for i := 0; i < len(v); i++ {
		v[i].ResetDerivatives()
	}
}
func (v DenseBareRealVector) ReverseOrder() {
	n := len(v)
	for i := 0; i < n/2; i++ {
		v[i], v[n-1-i] = v[n-1-i], v[i]
	}
}
func (v DenseBareRealVector) Slice(i, j int) Vector {
	return v[i:j]
}
func (v DenseBareRealVector) Append(w DenseBareRealVector) DenseBareRealVector {
	return append(v, w...)
}
func (v DenseBareRealVector) AppendScalar(scalars ...Scalar) Vector {
	for _, scalar := range scalars {
		switch s := scalar.(type) {
		case *BareReal:
			v = append(v, *s)
		default:
			v = append(v, *s.ConvertType(BareRealType).(*BareReal))
		}
	}
	return v
}
func (v DenseBareRealVector) AppendVector(w_ Vector) Vector {
	switch w := w_.(type) {
	case DenseBareRealVector:
		return append(v, w...)
	default:
		for i := 0; i < w.Dim(); i++ {
			v = append(v, *w.At(i).ConvertType(BareRealType).(*BareReal))
		}
		return v
	}
}
func (v DenseBareRealVector) Swap(i, j int) {
	v[i], v[j] = v[j], v[i]
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Map(f func(Scalar)) {
	for i := 0; i < len(v); i++ {
		f(&v[i])
	}
}
func (v DenseBareRealVector) MapSet(f func(ConstScalar) Scalar) {
	for i := 0; i < len(v); i++ {
		v[i].Set(f(v.ConstAt(i)))
	}
}
func (v DenseBareRealVector) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
	for i := 0; i < len(v); i++ {
		r = f(r, v.ConstAt(i))
	}
	return r
}
func (v DenseBareRealVector) ElementType() ScalarType {
	return BareRealType
}
func (v DenseBareRealVector) Variables(order int) error {
	for i, _ := range v {
		if err := v[i].SetVariable(i, len(v), order); err != nil {
			return err
		}
	}
	return nil
}

/* permutations
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) Permute(pi []int) error {
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
type sortDenseBareRealVectorByValue DenseBareRealVector

func (v sortDenseBareRealVectorByValue) Len() int           { return len(v) }
func (v sortDenseBareRealVectorByValue) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v sortDenseBareRealVectorByValue) Less(i, j int) bool { return v[i].GetValue() < v[j].GetValue() }
func (v DenseBareRealVector) Sort(reverse bool) {
	if reverse {
		sort.Sort(sort.Reverse(sortDenseBareRealVectorByValue(v)))
	} else {
		sort.Sort(sortDenseBareRealVectorByValue(v))
	}
}

/* type conversion
 * -------------------------------------------------------------------------- */
func (v DenseBareRealVector) AsMatrix(n, m int) Matrix {
	return v.ToDenseBareRealMatrix(n, m)
}
func (v DenseBareRealVector) ToDenseBareRealMatrix(n, m int) *DenseBareRealMatrix {
	if n*m != len(v) {
		panic("Matrix dimension does not fit input vector!")
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
func (v DenseBareRealVector) String() string {
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
func (v DenseBareRealVector) Table() string {
	var buffer bytes.Buffer
	for i, _ := range v {
		if i != 0 {
			buffer.WriteString(" ")
		}
		buffer.WriteString(v[i].String())
	}
	return buffer.String()
}
func (v DenseBareRealVector) Export(filename string) error {
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
func (v *DenseBareRealVector) Import(filename string) error {
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
	*v = DenseBareRealVector{}
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
			*v = append(*v, *NewBareReal(value))
		}
	}
	return nil
}

/* json
 * -------------------------------------------------------------------------- */
func (obj DenseBareRealVector) MarshalJSON() ([]byte, error) {
	r := []BareReal{}
	r = obj
	return json.MarshalIndent(r, "", "  ")
}
func (obj *DenseBareRealVector) UnmarshalJSON(data []byte) error {
	r := []BareReal{}
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	*obj = nilDenseBareRealVector(len(r))
	for i := 0; i < len(r); i++ {
		(*obj)[i] = r[i]
	}
	return nil
}

/* iterator
 * -------------------------------------------------------------------------- */
type DenseBareRealVectorIterator struct {
	v DenseBareRealVector
	i int
}

func (obj *DenseBareRealVectorIterator) Get() Scalar {
	return obj.GET()
}
func (obj *DenseBareRealVectorIterator) GetConst() ConstScalar {
	return obj.GET()
}
func (obj *DenseBareRealVectorIterator) GetValue() float64 {
	return obj.GET().GetValue()
}
func (obj *DenseBareRealVectorIterator) GET() *BareReal {
	return &obj.v[obj.i]
}
func (obj *DenseBareRealVectorIterator) Ok() bool {
	return obj.i < len(obj.v)
}
func (obj *DenseBareRealVectorIterator) Next() {
	obj.i++
}
func (obj *DenseBareRealVectorIterator) Index() int {
	return obj.i
}
func (obj *DenseBareRealVectorIterator) Clone() *DenseBareRealVectorIterator {
	return &DenseBareRealVectorIterator{obj.v, obj.i}
}
func (obj *DenseBareRealVectorIterator) CloneConstIterator() VectorConstIterator {
	return &DenseBareRealVectorIterator{obj.v, obj.i}
}
func (obj *DenseBareRealVectorIterator) CloneIterator() VectorIterator {
	return &DenseBareRealVectorIterator{obj.v, obj.i}
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseBareRealVectorJointIterator struct {
	it1 *DenseBareRealVectorIterator
	it2 VectorConstIterator
	idx int
	s1  *BareReal
	s2  ConstScalar
}

func (obj *DenseBareRealVectorJointIterator) Index() int {
	return obj.idx
}
func (obj *DenseBareRealVectorJointIterator) Ok() bool {
	return !(obj.s1 == nil || obj.s1.GetValue() == 0.0) ||
		!(obj.s2 == nil || obj.s2.GetValue() == 0.0)
}
func (obj *DenseBareRealVectorJointIterator) Next() {
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
func (obj *DenseBareRealVectorJointIterator) Get() (Scalar, ConstScalar) {
	return obj.GET()
}
func (obj *DenseBareRealVectorJointIterator) GetConst() (ConstScalar, ConstScalar) {
	return obj.GET()
}
func (obj *DenseBareRealVectorJointIterator) GetValue() (float64, float64) {
	a, b := obj.GET()
	return a.GetValue(), b.GetValue()
}
func (obj *DenseBareRealVectorJointIterator) GET() (*BareReal, ConstScalar) {
	if obj.s1 == nil {
		return nil, obj.s2
	} else {
		return obj.s1, obj.s2
	}
}
func (obj *DenseBareRealVectorJointIterator) Clone() *DenseBareRealVectorJointIterator {
	r := DenseBareRealVectorJointIterator{}
	r.it1 = obj.it1.Clone()
	r.it2 = obj.it2.CloneConstIterator()
	r.idx = obj.idx
	r.s1 = obj.s1
	r.s2 = obj.s2
	return &r
}
func (obj *DenseBareRealVectorJointIterator) CloneConstJointIterator() VectorConstJointIterator {
	return obj.Clone()
}
func (obj *DenseBareRealVectorJointIterator) CloneJointIterator() VectorJointIterator {
	return obj.Clone()
}

/* joint iterator
 * -------------------------------------------------------------------------- */
type DenseBareRealVectorJointIterator_ struct {
	it1 *DenseBareRealVectorIterator
	it2 *DenseBareRealVectorIterator
	idx int
	s1  *BareReal
	s2  *BareReal
}

func (obj *DenseBareRealVectorJointIterator_) Index() int {
	return obj.idx
}
func (obj *DenseBareRealVectorJointIterator_) Ok() bool {
	return obj.s1 != nil || obj.s2 != nil
}
func (obj *DenseBareRealVectorJointIterator_) Next() {
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
func (obj *DenseBareRealVectorJointIterator_) GET() (*BareReal, *BareReal) {
	return obj.s1, obj.s2
}
