/* Copyright (C) 2015 Philipp Benner
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

package autodiff

/* -------------------------------------------------------------------------- */

import "fmt"
import "encoding/json"
import "reflect"
import "math"

/* -------------------------------------------------------------------------- */

type BareReal float64

/* register scalar type
 * -------------------------------------------------------------------------- */

var BareRealType ScalarType = NewBareReal(0.0).Type()

func init() {
	f := func(value float64) Scalar { return NewBareReal(value) }
	RegisterScalar(BareRealType, f)
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewBareReal(v float64) *BareReal {
	r := BareReal(v)
	return &r
}

func NullBareReal() *BareReal {
	r := BareReal(0.0)
	return &r
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Clone() *BareReal {
	return NewBareReal(float64(*a))
}

func (a *BareReal) CloneScalar() Scalar {
	return a.Clone()
}

func (a *BareReal) Type() ScalarType {
	return reflect.TypeOf(a)
}

func (a *BareReal) ConvertType(t ScalarType) Scalar {
	switch t {
	case RealType:
		return NewReal(a.GetValue())
	case BareRealType:
		return a
	default:
		panic(fmt.Sprintf("cannot convert `BareReal' to type `%v'", t))
	}
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a BareReal) String() string {
	return fmt.Sprintf("%e", a.GetValue())
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Alloc(n, order int) {
}

func (c *BareReal) AllocForOne(a ConstScalar) {
}

func (c *BareReal) AllocForTwo(a, b ConstScalar) {
}

/* read access
 * -------------------------------------------------------------------------- */

func (a BareReal) GetOrder() int {
	return 0
}

func (a BareReal) GetValue() float64 {
	return float64(a)
}

func (a BareReal) GetLogValue() float64 {
	return math.Log(a.GetValue())
}

func (a BareReal) GetDerivative(i int) float64 {
	return 0.0
}

func (a BareReal) GetHessian(i, j int) float64 {
	return 0.0
}

func (a BareReal) GetN() int {
	return 0
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BareReal) Reset() {
	*a = 0.0
}

func (a *BareReal) ResetDerivatives() {
}

func (a *BareReal) Set(b ConstScalar) {
	*a = BareReal(b.GetValue())
}

func (a *BareReal) SET(b *BareReal) {
	*a = *b
}

func (a *BareReal) SetValue(v float64) {
	*a = BareReal(v)
}

func (a *BareReal) setValue(v float64) {
	*a = BareReal(v)
}

func (a *BareReal) SetDerivative(i int, v float64) {
}

func (a *BareReal) SetHessian(i, j int, v float64) {
}

func (a *BareReal) SetVariable(i, n, order int) error {
	return fmt.Errorf("BareReal cannot be used as a variable")
}

func (a *BareReal) SetN(n int) {
}

/* json
 * -------------------------------------------------------------------------- */

func (obj *BareReal) MarshalJSON() ([]byte, error) {
	return json.Marshal(*obj)
}

func (obj *BareReal) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, (*float64)(obj))
}
