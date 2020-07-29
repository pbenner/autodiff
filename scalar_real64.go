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
type Real64 struct {
  Value float64
  Order int
  Derivative []float64
  Hessian [][]float64
  N int
}
/* register scalar type
 * -------------------------------------------------------------------------- */
var Real64Type ScalarType = NewReal64(0.0).Type()
func init() {
  f := func(value float64) Scalar { return NewReal64(float64(value)) }
  RegisterScalar(Real64Type, f)
}
/* constructors
 * -------------------------------------------------------------------------- */
// Create a new real constant or variable.
func NewReal64(v float64) *Real64 {
  s := Real64{}
  s.Value = v
  s.Order = 0
  s.N = 0
  return &s
}
func NullReal64() *Real64 {
  return NewReal64(0.0)
}
/* -------------------------------------------------------------------------- */
func (a *Real64) Clone() *Real64 {
  r := NewReal64(0.0)
  r.Set(a)
  return r
}
func (a *Real64) CloneConstScalar() ConstScalar {
  return a.Clone()
}
func (a *Real64) CloneScalar() Scalar {
  return a.Clone()
}
func (a *Real64) CloneMagicScalar() MagicScalar {
  return a.Clone()
}
/* -------------------------------------------------------------------------- */
func (a *Real64) Type() ScalarType {
  return reflect.TypeOf(a)
}
/* type conversion
 * -------------------------------------------------------------------------- */
func (a *Real64) ConvertScalar(t ScalarType) Scalar {
  switch t {
  case Real64Type:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return a
  }
}
func (a *Real64) ConvertMagicScalar(t ScalarType) MagicScalar {
  switch t {
  case Real64Type:
    return a
  default:
    r := NullScalar(t)
    r.Set(a)
    return a
  }
}
func (a *Real64) ConvertConstScalar(t ScalarType) ConstScalar {
  switch t {
  case Real64Type:
    return a
  default:
    return NewConstScalar(t, a.GetFloat64())
  }
}
/* stringer
 * -------------------------------------------------------------------------- */
func (a *Real64) String() string {
  return fmt.Sprintf("%v", a.GetFloat64())
}
/* -------------------------------------------------------------------------- */
// Allocate memory for derivatives of n variables.
func (a *Real64) Alloc(n, order int) {
  if a.N != n || a.Order != order {
    a.N = n
    a.Order = order
    // allocate gradient if requested
    if a.Order >= 1 {
      a.Derivative = make([]float64, n)
      // allocate Hessian if requested
      if a.Order >= 2 {
        a.Hessian = make([][]float64, n)
        for i := 0; i < n; i++ {
          a.Hessian[i] = make([]float64, n)
        }
      } else {
        a.Hessian = nil
      }
    } else {
      a.Derivative = nil
    }
  }
}
// Allocate memory for the results of mathematical operations on
// the given variables.
func (c *Real64) AllocForOne(a ConstScalar) {
  c.Alloc(a.GetN(), a.GetOrder())
}
func (c *Real64) AllocForTwo(a, b ConstScalar) {
  c.Alloc(iMax(a.GetN(), b.GetN()), iMax(a.GetOrder(), b.GetOrder()))
}
/* read access
 * -------------------------------------------------------------------------- */
func (a *Real64) GetInt8() int8 {
  return int8(a.Value)
}
func (a *Real64) GetInt16() int16 {
  return int16(a.Value)
}
func (a *Real64) GetInt32() int32 {
  return int32(a.Value)
}
func (a *Real64) GetInt64() int64 {
  return int64(a.Value)
}
func (a *Real64) GetInt() int {
  return int(a.Value)
}
func (a *Real64) GetFloat32() float32 {
  return float32(a.Value)
}
func (a *Real64) GetFloat64() float64 {
  return float64(a.Value)
}
// Indicates the maximal order of derivatives that are computed for this
// variable. `0' means no derivatives, `1' only the first derivative, and
// `2' the first and second derivative.
func (a *Real64) GetOrder() int {
  return a.Order
}
// Returns the value of the variable on log scale.
func (a *Real64) GetLogValue() float64 {
  return math.Log(float64(a.Value))
}
// Returns the derivative of the ith variable.
func (a *Real64) GetDerivative(i int) float64 {
  if a.Order >= 1 {
    return float64(a.Derivative[i])
  } else {
    return 0.0
  }
}
func (a *Real64) GetHessian(i, j int) float64 {
  if a.Order >= 2 {
    return float64(a.Hessian[i][j])
  } else {
    return 0.0
  }
}
// Number of variables for which derivates are stored.
func (a *Real64) GetN() int {
  return a.N
}
/* write access
 * -------------------------------------------------------------------------- */
func (a *Real64) Reset() {
  a.Value = 0.0
  a.ResetDerivatives()
}
// Set the state to b. This includes the value and all derivatives.
func (a *Real64) Set(b ConstScalar) {
  a.Value = b.GetFloat64()
  a.Order = b.GetOrder()
  a.Alloc(b.GetN(), b.GetOrder())
  if a.Order >= 1 {
    for i := 0; i < b.GetN(); i++ {
      a.Derivative[i] = float64(b.GetDerivative(i))
    }
    if a.Order >= 2 {
      for i := 0; i < b.GetN(); i++ {
        for j := 0; j < b.GetN(); j++ {
          a.Hessian[i][j] = float64(b.GetHessian(i, j))
        }
      }
    }
  }
}
func (a *Real64) SET(b *Real64) {
  a.Value = b.GetFloat64()
  a.Order = b.GetOrder()
  a.Alloc(b.GetN(), b.GetOrder())
  if a.Order >= 1 {
    for i := 0; i < b.GetN(); i++ {
      a.Derivative[i] = float64(b.GetDerivative(i))
    }
    if a.Order >= 2 {
      for i := 0; i < b.GetN(); i++ {
        for j := 0; j < b.GetN(); j++ {
          a.Hessian[i][j] = float64(b.GetHessian(i, j))
        }
      }
    }
  }
}
// Set the value of the variable. All derivatives are reset to zero.
func (a *Real64) SetInt8(v int8) {
  a.setInt8(v)
  a.ResetDerivatives()
}
func (a *Real64) setInt8(v int8) {
  a.Value = float64(v)
}
func (a *Real64) SetInt16(v int16) {
  a.setInt16(v)
  a.ResetDerivatives()
}
func (a *Real64) setInt16(v int16) {
  a.Value = float64(v)
}
func (a *Real64) SetInt32(v int32) {
  a.setInt32(v)
  a.ResetDerivatives()
}
func (a *Real64) setInt32(v int32) {
  a.Value = float64(v)
}
func (a *Real64) SetInt64(v int64) {
  a.setInt64(v)
  a.ResetDerivatives()
}
func (a *Real64) setInt64(v int64) {
  a.Value = float64(v)
}
func (a *Real64) SetInt(v int) {
  a.setInt(v)
  a.ResetDerivatives()
}
func (a *Real64) setInt(v int) {
  a.Value = float64(v)
}
func (a *Real64) SetFloat32(v float32) {
  a.setFloat32(v)
  a.ResetDerivatives()
}
func (a *Real64) setFloat32(v float32) {
  a.Value = float64(v)
}
func (a *Real64) SetFloat64(v float64) {
  a.setFloat64(v)
  a.ResetDerivatives()
}
func (a *Real64) setFloat64(v float64) {
  a.Value = float64(v)
}
/* magic write access
 * -------------------------------------------------------------------------- */
func (a *Real64) ResetDerivatives() {
  if a.Order >= 1 {
    for i := 0; i < a.N; i++ {
      a.Derivative[i] = 0.0
    }
    if a.Order >= 2 {
      for i := 0; i < a.N; i++ {
        for j := 0; j < a.N; j++ {
          a.Hessian[i][j] = 0.0
        }
      }
    }
  }
}
// Set the derivative of the ith variable to v.
func (a *Real64) SetDerivative(i int, v float64) {
  a.Derivative[i] = float64(v)
}
func (a *Real64) SetHessian(i, j int, v float64) {
  a.Hessian[i][j] = float64(v)
}
// Allocate memory for n variables and set the derivative
// of the ith variable to 1 (initial value).
func (a *Real64) SetVariable(i, n, order int) error {
  if order > 2 {
    return fmt.Errorf("order `%d' not supported by this type", order)
  }
  a.Alloc(n, order)
  if order > 0 {
    a.Derivative[i] = 1
  }
  return nil
}
/* -------------------------------------------------------------------------- */
func (a *Real64) nullScalar() bool {
  if a == nil {
    return true
  }
  if a.Value != 0 {
    return false
  }
  if a.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      if v := a.GetDerivative(i); v != 0.0 {
        return false
      }
    }
  }
  if a.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      for j := 0; j < a.GetN(); j++ {
        if v := a.GetHessian(i, j); v != 0.0 {
          return false
        }
      }
    }
  }
  return true
}
/* json
 * -------------------------------------------------------------------------- */
func (obj *Real64) MarshalJSON() ([]byte, error) {
  t1 := false
  t2 := false
  if obj.Order > 0 && obj.N > 0 {
    // check for non-zero derivatives
    for i := 0; !t1 && i < obj.GetN(); i++ {
      if obj.Derivative[i] != 0.0 {
        t1 = true
      }
    }
    if obj.Order > 1 {
      // check for non-zero second derivatives
      for i := 0; !t2 && i < obj.GetN(); i++ {
        for j := 0; !t2 && j < obj.GetN(); j++ {
          if obj.GetHessian(i, j) != 0.0 {
            t2 = true
          }
        }
      }
    }
  }
  if t1 && t2 {
    r := struct{Value float64; Derivative []float64; Hessian [][]float64}{
      obj.Value, obj.Derivative, obj.Hessian}
    return json.Marshal(r)
  } else
  if t1 && !t2 {
    r := struct{Value float64; Derivative []float64}{
      obj.Value, obj.Derivative}
    return json.Marshal(r)
  } else
  if !t1 && t2 {
    r := struct{Value float64; Hessian [][]float64}{
      obj.Value, obj.Hessian}
    return json.Marshal(r)
  } else {
    return json.Marshal(obj.Value)
  }
}
func (obj *Real64) UnmarshalJSON(data []byte) error {
  r := struct{Value float64; Derivative []float64; Hessian [][]float64}{}
  if err := json.Unmarshal(data, &r); err == nil {
    obj.Value = r.Value
    if len(r.Derivative) != 0 && len(r.Hessian) != 0 {
      if len(r.Derivative) != len(r.Derivative) {
        return fmt.Errorf("invalid json scalar representation")
      }
      obj.Alloc(len(r.Derivative), 2)
      obj.Derivative = r.Derivative
      obj.Hessian = r.Hessian
    } else
    if len(r.Derivative) != 0 && len(r.Hessian) == 0 {
      obj.Alloc(len(r.Derivative), 1)
      obj.Derivative = r.Derivative
    } else
    if len(r.Derivative) == 0 && len(r.Hessian) != 0 {
      obj.Alloc(len(r.Derivative), 2)
      obj.Hessian = r.Hessian
    }
    return nil
  } else {
    return json.Unmarshal(data, &obj.Value)
  }
}
