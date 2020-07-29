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
type ConstFloat64 float64
/* register scalar type
 * -------------------------------------------------------------------------- */
var ConstFloat64Type ScalarType = NewConstFloat64(0.0).Type()
func init() {
  f := func(value float64) ConstScalar { return NewConstFloat64(float64(value)) }
  RegisterConstScalar(ConstFloat64Type, f)
}
/* constructors
 * -------------------------------------------------------------------------- */
func NewConstFloat64(v float64) ConstFloat64 {
  return ConstFloat64(v)
}
func NullConstFloat64() ConstFloat64 {
  return ConstFloat64(0.0)
}
/* -------------------------------------------------------------------------- */
func (a ConstFloat64) Clone() ConstFloat64 {
  return ConstFloat64(a)
}
func (a ConstFloat64) CloneConstScalar() ConstScalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a ConstFloat64) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a ConstFloat64) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case ConstFloat64Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
/* stringer
 * -------------------------------------------------------------------------- */
func (a ConstFloat64) String() string {
  return fmt.Sprintf("%v", a.GetFloat64())
}
/* read access
 * -------------------------------------------------------------------------- */
func (a ConstFloat64) GetInt8() int8 {
  return int8(a)
}
func (a ConstFloat64) GetInt16() int16 {
  return int16(a)
}
func (a ConstFloat64) GetInt32() int32 {
  return int32(a)
}
func (a ConstFloat64) GetInt64() int64 {
  return int64(a)
}
func (a ConstFloat64) GetInt() int {
  return int(a)
}
func (a ConstFloat64) GetFloat32() float32 {
  return float32(a)
}
func (a ConstFloat64) GetFloat64() float64 {
  return float64(a)
}
func (a ConstFloat64) GetOrder() int {
  return 0
}
func (a ConstFloat64) GetDerivative(i int) float64 {
  return 0.0
}
func (a ConstFloat64) GetHessian(i, j int) float64 {
  return 0.0
}
func (a ConstFloat64) GetN() int {
  return 0
}
/* json
 * -------------------------------------------------------------------------- */
func (obj ConstFloat64) MarshalJSON() ([]byte, error) {
  return json.Marshal(obj)
}
/* math
 * -------------------------------------------------------------------------- */
func (a ConstFloat64) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a ConstFloat64) Greater(b ConstScalar) bool {
  return a.GetFloat64() > b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a ConstFloat64) Smaller(b ConstScalar) bool {
  return a.GetFloat64() < b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a ConstFloat64) Sign() int {
  if a.GetFloat64() < float64(0) {
    return -1
  }
  if a.GetFloat64() > float64(0) {
    return 1
  }
  return 0
}
