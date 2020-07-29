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
//import "fmt"
import "math"
//import "github.com/pbenner/autodiff/special"
/* -------------------------------------------------------------------------- */
func (a Int32) EQUALS(b Int32, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a Int32) GREATER(b Int32) bool {
  return a.GetInt32() > b.GetInt32()
}
/* -------------------------------------------------------------------------- */
func (a Int32) SMALLER(b Int32) bool {
  return a.GetInt32() < b.GetInt32()
}
/* -------------------------------------------------------------------------- */
func (a Int32) SIGN() int {
  if a.GetInt32() < int32(0) {
    return -1
  }
  if a.GetInt32() > int32(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r Int32) MIN(a, b Int32) Scalar {
  if a.GetInt32() < b.GetInt32() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r Int32) MAX(a, b Int32) Scalar {
  if a.GetInt32() > b.GetInt32() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c Int32) ABS(a Int32) Scalar {
  if c.Sign() == -1 {
    c.NEG(a)
  } else {
    c.SET(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) NEG(a Int32) Int32 {
  x := a.GetInt32()
  c.SetInt32(-x)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) ADD(a, b Int32) Int32 {
  x := a.GetInt32()
  y := b.GetInt32()
  c.SetInt32(x+y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) SUB(a, b Int32) Int32 {
  x := a.GetInt32()
  y := b.GetInt32()
  c.SetInt32(x-y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) MUL(a, b Int32) Int32 {
  x := a.GetInt32()
  y := b.GetInt32()
  c.SetInt32(x*y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) DIV(a, b Int32) Int32 {
  x := a.GetInt32()
  y := b.GetInt32()
  c.SetInt32(x/y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) LOGADD(a, b, t Int32) Int32 {
  if a.GREATER(b) {
    // swap
    a, b = b, a
  }
  if math.IsInf(a.GetFloat64(), 0) {
    // cases:
    //  i) a = -Inf and b >= a    => c = b
    // ii) a =  Inf and b  = Inf  => c = Inf
    c.SET(b)
    return c
  }
  t.SUB(a, b)
  t.EXP(t)
  t.LOG1P(t)
  c.ADD(t, b)
  return c
}
func (c Int32) LOGSUB(a, b, t Int32) Int32 {
  if math.IsInf(b.GetFloat64(), -1) {
    c.SET(a)
    return c
  }
  t.SUB(b, a)
  t.EXP(t)
  t.NEG(t)
  t.LOG1P(t)
  c.ADD(t, a)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) POW(a, k Int32) Int32 {
  x := a.GetFloat64()
  y := k.GetFloat64()
  c.SetFloat64(math.Pow(x, y))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) SQRT(a Int32) Int32 {
  x := a.GetFloat64()
  c.SetFloat64(math.Sqrt(x))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int32) EXP(a Int32) Int32 {
  x := a.GetFloat64()
  c.SetFloat64(math.Exp(x))
  return c
}
func (c Int32) LOG(a Int32) Int32 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log(x))
  return c
}
func (c Int32) LOG1P(a Int32) Int32 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log1p(x))
  return c
}
