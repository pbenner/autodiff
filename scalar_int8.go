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
import "reflect"
/* -------------------------------------------------------------------------- */
type Int8 struct {
  ptr *int8
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewInt8(v int8) Int8 {
  return Int8{&v}
}
func NullInt8() Int8 {
  v := int8(0.0)
  return Int8{&v}
}
/* register scalar type
 * -------------------------------------------------------------------------- */
var Int8Type ScalarType = (Int8{}).Type()
func init() {
  f := func(value float64) Scalar { return NewInt8(int8(value)) }
  RegisterScalar(Int8Type, f)
}
/* -------------------------------------------------------------------------- */
func (a Int8) Clone() Int8 {
  r := NewInt8(0.0)
  r.Set(a)
  return r
}
func (a Int8) CloneConstScalar() ConstScalar {
  return a.Clone()
}
func (a Int8) CloneScalar() Scalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a Int8) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* -------------------------------------------------------------------------- */
func (a Int8) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case Int8Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
func (a Int8) ConvertScalar(t ScalarType) Scalar {
  switch t {
  case Int8Type:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return r
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a Int8) String() string {
  return fmt.Sprintf("%v", a.GetInt8())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a Int8) GetInt8() int8 {
  return int8(*a.ptr)
}
func (a Int8) GetInt16() int16 {
  return int16(*a.ptr)
}
func (a Int8) GetInt32() int32 {
  return int32(*a.ptr)
}
func (a Int8) GetInt64() int64 {
  return int64(*a.ptr)
}
func (a Int8) GetInt() int {
  return int(*a.ptr)
}
func (a Int8) GetFloat32() float32 {
  return float32(*a.ptr)
}
func (a Int8) GetFloat64() float64 {
  return float64(*a.ptr)
}
func (a Int8) GetOrder() int {
  return 0
}
func (a Int8) GetDerivative(i int) float64 {
  return 0.0
}
func (a Int8) GetHessian(i, j int) float64 {
  return 0.0
}
func (a Int8) GetN() int {
  return 0
}
/* write access
 * -------------------------------------------------------------------------- */
func (a Int8) Reset() {
  *a.ptr = 0.0
}
// Set the state to b. This includes the value and all derivatives.
func (a Int8) Set(b ConstScalar) {
  *a.ptr = b.GetInt8()
}
func (a Int8) SET(b Int8) {
  *a.ptr = *b.ptr
}
// Set the value of the variable. All derivatives are reset to zero.
func (a Int8) SetInt8(v int8) {
  a.setInt8(v)
}
func (a Int8) setInt8(v int8) {
  *a.ptr = int8(v)
}
func (a Int8) SetInt16(v int16) {
  a.setInt16(v)
}
func (a Int8) setInt16(v int16) {
  *a.ptr = int8(v)
}
func (a Int8) SetInt32(v int32) {
  a.setInt32(v)
}
func (a Int8) setInt32(v int32) {
  *a.ptr = int8(v)
}
func (a Int8) SetInt64(v int64) {
  a.setInt64(v)
}
func (a Int8) setInt64(v int64) {
  *a.ptr = int8(v)
}
func (a Int8) SetInt(v int) {
  a.setInt(v)
}
func (a Int8) setInt(v int) {
  *a.ptr = int8(v)
}
func (a Int8) SetFloat32(v float32) {
  a.setFloat32(v)
}
func (a Int8) setFloat32(v float32) {
  *a.ptr = int8(v)
}
func (a Int8) SetFloat64(v float64) {
  a.setFloat64(v)
}
func (a Int8) setFloat64(v float64) {
  *a.ptr = int8(v)
}
/* -------------------------------------------------------------------------- */
func (a Int8) nullScalar() bool {
  if a.ptr == nil {
    return true
  }
  if *a.ptr != 0 {
    return false
  }
  return true
}
/* json
 * -------------------------------------------------------------------------- */
func (obj Int8) MarshalJSON() ([]byte, error) {
  return json.Marshal(*obj.ptr)
}
func (obj Int8) UnmarshalJSON(data []byte) error {
  return json.Unmarshal(data, (*int8)(obj.ptr))
}
