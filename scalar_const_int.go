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
import "encoding/json"
import "math"
import "reflect"
/* -------------------------------------------------------------------------- */
type ConstInt int
/* register scalar type
 * -------------------------------------------------------------------------- */
var ConstIntType ScalarType = NewConstInt(0.0).Type()
func init() {
  f := func(value float64) ConstScalar { return NewConstInt(int(value)) }
  RegisterConstScalar(ConstIntType, f)
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewConstInt(v int) ConstInt {
  return ConstInt(v)
}
func NullConstInt() ConstInt {
  return ConstInt(0.0)
}
/* -------------------------------------------------------------------------- */
func (a ConstInt) Clone() ConstInt {
  return ConstInt(a)
}
func (a ConstInt) CloneConstScalar() ConstScalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a ConstInt) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case ConstIntType:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
/* stringer
 * -------------------------------------------------------------------------- */
func (a ConstInt) String() string {
  return fmt.Sprintf("%v", a.GetInt())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a ConstInt) GetInt8() int8 {
  return int8(a)
}
func (a ConstInt) GetInt16() int16 {
  return int16(a)
}
func (a ConstInt) GetInt32() int32 {
  return int32(a)
}
func (a ConstInt) GetInt64() int64 {
  return int64(a)
}
func (a ConstInt) GetInt() int {
  return int(a)
}
func (a ConstInt) GetFloat32() float32 {
  return float32(a)
}
func (a ConstInt) GetFloat64() float64 {
  return float64(a)
}
func (a ConstInt) GetOrder() int {
  return 0
}
func (a ConstInt) GetDerivative(i int) float64 {
  return 0.0
}
func (a ConstInt) GetHessian(i, j int) float64 {
  return 0.0
}
func (a ConstInt) GetN() int {
  return 0
}
/* json
 * -------------------------------------------------------------------------- */
func (obj ConstInt) MarshalJSON() ([]byte, error) {
  return json.Marshal(obj)
}
/* math
 * -------------------------------------------------------------------------- */
func (a ConstInt) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a ConstInt) Greater(b ConstScalar) bool {
  return a.GetInt() > b.GetInt()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt) Smaller(b ConstScalar) bool {
  return a.GetInt() < b.GetInt()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt) Sign() int {
  if a.GetInt() < int(0) {
    return -1
  }
  if a.GetInt() > int(0) {
    return 1
  }
  return 0
}
