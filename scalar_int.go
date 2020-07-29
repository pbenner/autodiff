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
type Int struct {
  ptr *int
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewInt(v int) Int {
  return Int{&v}
}
func NullInt() Int {
  v := int(0.0)
  return Int{&v}
}
/* register scalar type
 * -------------------------------------------------------------------------- */
var IntType ScalarType = (Int{}).Type()
func init() {
  f := func(value float64) Scalar { return NewInt(int(value)) }
  RegisterScalar(IntType, f)
}
/* -------------------------------------------------------------------------- */
func (a Int) Clone() Int {
  r := NewInt(0.0)
  r.Set(a)
  return r
}
func (a Int) CloneConstScalar() ConstScalar {
  return a.Clone()
}
func (a Int) CloneScalar() Scalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a Int) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* -------------------------------------------------------------------------- */
func (a Int) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case IntType:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
func (a Int) ConvertScalar(t ScalarType) Scalar {
  switch t {
  case IntType:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return r
  }
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a Int) String() string {
  return fmt.Sprintf("%v", a.GetInt())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a Int) GetInt8() int8 {
  return int8(*a.ptr)
}
func (a Int) GetInt16() int16 {
  return int16(*a.ptr)
}
func (a Int) GetInt32() int32 {
  return int32(*a.ptr)
}
func (a Int) GetInt64() int64 {
  return int64(*a.ptr)
}
func (a Int) GetInt() int {
  return int(*a.ptr)
}
func (a Int) GetFloat32() float32 {
  return float32(*a.ptr)
}
func (a Int) GetFloat64() float64 {
  return float64(*a.ptr)
}
func (a Int) GetOrder() int {
  return 0
}
func (a Int) GetDerivative(i int) float64 {
  return 0.0
}
func (a Int) GetHessian(i, j int) float64 {
  return 0.0
}
func (a Int) GetN() int {
  return 0
}
/* write access
 * -------------------------------------------------------------------------- */
func (a Int) Reset() {
  *a.ptr = 0.0
}
// Set the state to b. This includes the value and all derivatives.
func (a Int) Set(b ConstScalar) {
  *a.ptr = b.GetInt()
}
func (a Int) SET(b Int) {
  *a.ptr = *b.ptr
}
// Set the value of the variable. All derivatives are reset to zero.
func (a Int) SetInt8(v int8) {
  a.setInt8(v)
}
func (a Int) setInt8(v int8) {
  *a.ptr = int(v)
}
func (a Int) SetInt16(v int16) {
  a.setInt16(v)
}
func (a Int) setInt16(v int16) {
  *a.ptr = int(v)
}
func (a Int) SetInt32(v int32) {
  a.setInt32(v)
}
func (a Int) setInt32(v int32) {
  *a.ptr = int(v)
}
func (a Int) SetInt64(v int64) {
  a.setInt64(v)
}
func (a Int) setInt64(v int64) {
  *a.ptr = int(v)
}
func (a Int) SetInt(v int) {
  a.setInt(v)
}
func (a Int) setInt(v int) {
  *a.ptr = int(v)
}
func (a Int) SetFloat32(v float32) {
  a.setFloat32(v)
}
func (a Int) setFloat32(v float32) {
  *a.ptr = int(v)
}
func (a Int) SetFloat64(v float64) {
  a.setFloat64(v)
}
func (a Int) setFloat64(v float64) {
  *a.ptr = int(v)
}
/* -------------------------------------------------------------------------- */
func (a Int) nullScalar() bool {
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
func (obj Int) MarshalJSON() ([]byte, error) {
  return json.Marshal(*obj.ptr)
}
func (obj Int) UnmarshalJSON(data []byte) error {
  return json.Unmarshal(data, (*int)(obj.ptr))
}
