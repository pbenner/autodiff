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
type Float64 struct {
  ptr *float64
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewFloat64(v float64) Float64 {
  return Float64{&v}
}
func NullFloat64() Float64 {
  v := float64(0.0)
  return Float64{&v}
}
/* register scalar type
 * -------------------------------------------------------------------------- */
var Float64Type ScalarType = (Float64{}).Type()
func init() {
  f := func(value float64) Scalar { return NewFloat64(float64(value)) }
  RegisterScalar(Float64Type, f)
}
/* -------------------------------------------------------------------------- */
func (a Float64) Clone() Float64 {
  r := NewFloat64(0.0)
  r.Set(a)
  return r
}
func (a Float64) CloneConstScalar() ConstScalar {
  return a.Clone()
}
func (a Float64) CloneScalar() Scalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a Float64) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* -------------------------------------------------------------------------- */
func (a Float64) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case Float64Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
func (a Float64) ConvertScalar(t ScalarType) Scalar {
  switch t {
  case Float64Type:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return r
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a Float64) String() string {
  return fmt.Sprintf("%v", a.GetFloat64())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a Float64) GetInt8() int8 {
  return int8(*a.ptr)
}
func (a Float64) GetInt16() int16 {
  return int16(*a.ptr)
}
func (a Float64) GetInt32() int32 {
  return int32(*a.ptr)
}
func (a Float64) GetInt64() int64 {
  return int64(*a.ptr)
}
func (a Float64) GetInt() int {
  return int(*a.ptr)
}
func (a Float64) GetFloat32() float32 {
  return float32(*a.ptr)
}
func (a Float64) GetFloat64() float64 {
  return float64(*a.ptr)
}
func (a Float64) GetOrder() int {
  return 0
}
func (a Float64) GetDerivative(i int) float64 {
  return 0.0
}
func (a Float64) GetHessian(i, j int) float64 {
  return 0.0
}
func (a Float64) GetN() int {
  return 0
}
/* write access
 * -------------------------------------------------------------------------- */
func (a Float64) Reset() {
  *a.ptr = 0.0
}
// Set the state to b. This includes the value and all derivatives.
func (a Float64) Set(b ConstScalar) {
  *a.ptr = b.GetFloat64()
}
func (a Float64) SET(b Float64) {
  *a.ptr = *b.ptr
}
// Set the value of the variable. All derivatives are reset to zero.
func (a Float64) SetInt8(v int8) {
  a.setInt8(v)
}
func (a Float64) setInt8(v int8) {
  *a.ptr = float64(v)
}
func (a Float64) SetInt16(v int16) {
  a.setInt16(v)
}
func (a Float64) setInt16(v int16) {
  *a.ptr = float64(v)
}
func (a Float64) SetInt32(v int32) {
  a.setInt32(v)
}
func (a Float64) setInt32(v int32) {
  *a.ptr = float64(v)
}
func (a Float64) SetInt64(v int64) {
  a.setInt64(v)
}
func (a Float64) setInt64(v int64) {
  *a.ptr = float64(v)
}
func (a Float64) SetInt(v int) {
  a.setInt(v)
}
func (a Float64) setInt(v int) {
  *a.ptr = float64(v)
}
func (a Float64) SetFloat32(v float32) {
  a.setFloat32(v)
}
func (a Float64) setFloat32(v float32) {
  *a.ptr = float64(v)
}
func (a Float64) SetFloat64(v float64) {
  a.setFloat64(v)
}
func (a Float64) setFloat64(v float64) {
  *a.ptr = float64(v)
}
/* -------------------------------------------------------------------------- */
func (a Float64) nullScalar() bool {
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
func (obj Float64) MarshalJSON() ([]byte, error) {
  return json.Marshal(*obj.ptr)
}
func (obj Float64) UnmarshalJSON(data []byte) error {
  return json.Unmarshal(data, (*float64)(obj.ptr))
}
