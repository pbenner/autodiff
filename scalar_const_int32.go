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
type ConstInt32 int32
/* register scalar type
 * -------------------------------------------------------------------------- */
var ConstInt32Type ScalarType = NewConstInt32(0.0).Type()
func init() {
  f := func(value float64) ConstScalar { return NewConstInt32(int32(value)) }
  RegisterConstScalar(ConstInt32Type, f)
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewConstInt32(v int32) ConstInt32 {
  return ConstInt32(v)
}
func NullConstInt32() ConstInt32 {
  return ConstInt32(0.0)
}
/* -------------------------------------------------------------------------- */
func (a ConstInt32) Clone() ConstInt32 {
  return ConstInt32(a)
}
func (a ConstInt32) CloneConstScalar() ConstScalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt32) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a ConstInt32) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case ConstInt32Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
/* stringer
 * -------------------------------------------------------------------------- */
func (a ConstInt32) String() string {
  return fmt.Sprintf("%v", a.GetInt32())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a ConstInt32) GetInt8() int8 {
  return int8(a)
}
func (a ConstInt32) GetInt16() int16 {
  return int16(a)
}
func (a ConstInt32) GetInt32() int32 {
  return int32(a)
}
func (a ConstInt32) GetInt64() int64 {
  return int64(a)
}
func (a ConstInt32) GetInt() int {
  return int(a)
}
func (a ConstInt32) GetFloat32() float32 {
  return float32(a)
}
func (a ConstInt32) GetFloat64() float64 {
  return float64(a)
}
func (a ConstInt32) GetOrder() int {
  return 0
}
func (a ConstInt32) GetDerivative(i int) float64 {
  return 0.0
}
func (a ConstInt32) GetHessian(i, j int) float64 {
  return 0.0
}
func (a ConstInt32) GetN() int {
  return 0
}
/* json
 * -------------------------------------------------------------------------- */
func (obj ConstInt32) MarshalJSON() ([]byte, error) {
  return json.Marshal(obj)
}
/* math
 * -------------------------------------------------------------------------- */
func (a ConstInt32) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a ConstInt32) Greater(b ConstScalar) bool {
  return a.GetInt32() > b.GetInt32()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt32) Smaller(b ConstScalar) bool {
  return a.GetInt32() < b.GetInt32()
}
/* -------------------------------------------------------------------------- */
func (a ConstInt32) Sign() int {
  if a.GetInt32() < int32(0) {
    return -1
  }
  if a.GetInt32() > int32(0) {
    return 1
  }
  return 0
}
