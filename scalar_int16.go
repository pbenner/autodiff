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
type Int16 struct {
  ptr *int16
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewInt16(v int16) Int16 {
  return Int16{&v}
}
func NullInt16() Int16 {
  v := int16(0.0)
  return Int16{&v}
}
/* register scalar type
 * -------------------------------------------------------------------------- */
var Int16Type ScalarType = (Int16{}).Type()
func init() {
  f := func(value float64) Scalar { return NewInt16(int16(value)) }
  RegisterScalar(Int16Type, f)
}
/* -------------------------------------------------------------------------- */
func (a Int16) Clone() Int16 {
  r := NewInt16(0.0)
  r.Set(a)
  return r
}
func (a Int16) CloneConstScalar() ConstScalar {
  return a.Clone()
}
func (a Int16) CloneScalar() Scalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a Int16) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* -------------------------------------------------------------------------- */
func (a Int16) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case Int16Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
func (a Int16) ConvertScalar(t ScalarType) Scalar {
  switch t {
  case Int16Type:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return r
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a Int16) String() string {
  return fmt.Sprintf("%v", a.GetInt16())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a Int16) GetInt8() int8 {
  return int8(*a.ptr)
}
func (a Int16) GetInt16() int16 {
  return int16(*a.ptr)
}
func (a Int16) GetInt32() int32 {
  return int32(*a.ptr)
}
func (a Int16) GetInt64() int64 {
  return int64(*a.ptr)
}
func (a Int16) GetInt() int {
  return int(*a.ptr)
}
func (a Int16) GetFloat32() float32 {
  return float32(*a.ptr)
}
func (a Int16) GetFloat64() float64 {
  return float64(*a.ptr)
}
func (a Int16) GetOrder() int {
  return 0
}
func (a Int16) GetDerivative(i int) float64 {
  return 0.0
}
func (a Int16) GetHessian(i, j int) float64 {
  return 0.0
}
func (a Int16) GetN() int {
  return 0
}
/* write access
 * -------------------------------------------------------------------------- */
func (a Int16) Reset() {
  *a.ptr = 0.0
}
// Set the state to b. This includes the value and all derivatives.
func (a Int16) Set(b ConstScalar) {
  *a.ptr = b.GetInt16()
}
func (a Int16) SET(b Int16) {
  *a.ptr = *b.ptr
}
// Set the value of the variable. All derivatives are reset to zero.
func (a Int16) SetInt8(v int8) {
  a.setInt8(v)
}
func (a Int16) setInt8(v int8) {
  *a.ptr = int16(v)
}
func (a Int16) SetInt16(v int16) {
  a.setInt16(v)
}
func (a Int16) setInt16(v int16) {
  *a.ptr = int16(v)
}
func (a Int16) SetInt32(v int32) {
  a.setInt32(v)
}
func (a Int16) setInt32(v int32) {
  *a.ptr = int16(v)
}
func (a Int16) SetInt64(v int64) {
  a.setInt64(v)
}
func (a Int16) setInt64(v int64) {
  *a.ptr = int16(v)
}
func (a Int16) SetInt(v int) {
  a.setInt(v)
}
func (a Int16) setInt(v int) {
  *a.ptr = int16(v)
}
func (a Int16) SetFloat32(v float32) {
  a.setFloat32(v)
}
func (a Int16) setFloat32(v float32) {
  *a.ptr = int16(v)
}
func (a Int16) SetFloat64(v float64) {
  a.setFloat64(v)
}
func (a Int16) setFloat64(v float64) {
  *a.ptr = int16(v)
}
/* -------------------------------------------------------------------------- */
func (a Int16) nullScalar() bool {
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
func (obj Int16) MarshalJSON() ([]byte, error) {
  return json.Marshal(*obj.ptr)
}
func (obj Int16) UnmarshalJSON(data []byte) error {
  return json.Unmarshal(data, (*int16)(obj.ptr))
}
