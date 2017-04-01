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
import "reflect"

/* -------------------------------------------------------------------------- */

type Real struct {
  BasicState
}

/* register scalar type
 * -------------------------------------------------------------------------- */

var RealType ScalarType = NewReal(0.0).Type()

func init() {
  f := func(value float64) Scalar { return NewReal(value) }
  RegisterScalar(RealType, f)
}

/* constructors
 * -------------------------------------------------------------------------- */

// Create a new real constant or variable.
func NewReal(v float64) *Real {
  s := Real{*NewBasicState(v)}
  return &s
}

func NullReal() *Real {
  s := Real{*NewBasicState(0.0)}
  return &s
}

/* -------------------------------------------------------------------------- */

func (a *Real) Clone() Scalar {
  r := NewReal(0.0)
  r.Copy(a)
  return r
}

func (a *Real) Type() ScalarType {
  return reflect.TypeOf(a)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a *Real) String() string {
  return fmt.Sprintf("%e", a.GetValue())
}
