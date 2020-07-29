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
func (a Int16) EQUALS(b Int16, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a Int16) GREATER(b Int16) bool {
  return a.GetInt16() > b.GetInt16()
}
/* -------------------------------------------------------------------------- */
func (a Int16) SMALLER(b Int16) bool {
  return a.GetInt16() < b.GetInt16()
}
/* -------------------------------------------------------------------------- */
func (a Int16) SIGN() int {
  if a.GetInt16() < int16(0) {
    return -1
  }
  if a.GetInt16() > int16(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r Int16) MIN(a, b Int16) Scalar {
  if a.GetInt16() < b.GetInt16() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r Int16) MAX(a, b Int16) Scalar {
  if a.GetInt16() > b.GetInt16() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c Int16) ABS(a Int16) Scalar {
  if c.Sign() == -1 {
    c.NEG(a)
  } else {
    c.SET(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) NEG(a Int16) Int16 {
  x := a.GetInt16()
  c.SetInt16(-x)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) ADD(a, b Int16) Int16 {
  x := a.GetInt16()
  y := b.GetInt16()
  c.SetInt16(x+y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) SUB(a, b Int16) Int16 {
  x := a.GetInt16()
  y := b.GetInt16()
  c.SetInt16(x-y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) MUL(a, b Int16) Int16 {
  x := a.GetInt16()
  y := b.GetInt16()
  c.SetInt16(x*y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) DIV(a, b Int16) Int16 {
  x := a.GetInt16()
  y := b.GetInt16()
  c.SetInt16(x/y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) LOGADD(a, b, t Int16) Int16 {
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
func (c Int16) LOGSUB(a, b, t Int16) Int16 {
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
func (c Int16) POW(a, k Int16) Int16 {
  x := a.GetFloat64()
  y := k.GetFloat64()
  c.SetFloat64(math.Pow(x, y))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) SQRT(a Int16) Int16 {
  x := a.GetFloat64()
  c.SetFloat64(math.Sqrt(x))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int16) EXP(a Int16) Int16 {
  x := a.GetFloat64()
  c.SetFloat64(math.Exp(x))
  return c
}
func (c Int16) LOG(a Int16) Int16 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log(x))
  return c
}
func (c Int16) LOG1P(a Int16) Int16 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log1p(x))
  return c
}
