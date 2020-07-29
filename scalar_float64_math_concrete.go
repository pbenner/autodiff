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
func (a Float64) EQUALS(b Float64, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a Float64) GREATER(b Float64) bool {
  return a.GetFloat64() > b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a Float64) SMALLER(b Float64) bool {
  return a.GetFloat64() < b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a Float64) SIGN() int {
  if a.GetFloat64() < float64(0) {
    return -1
  }
  if a.GetFloat64() > float64(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r Float64) MIN(a, b Float64) Scalar {
  if a.GetFloat64() < b.GetFloat64() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r Float64) MAX(a, b Float64) Scalar {
  if a.GetFloat64() > b.GetFloat64() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c Float64) ABS(a Float64) Scalar {
  if c.Sign() == -1 {
    c.NEG(a)
  } else {
    c.SET(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) NEG(a Float64) Float64 {
  x := a.GetFloat64()
  c.SetFloat64(-x)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) ADD(a, b Float64) Float64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  c.SetFloat64(x+y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) SUB(a, b Float64) Float64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  c.SetFloat64(x-y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) MUL(a, b Float64) Float64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  c.SetFloat64(x*y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) DIV(a, b Float64) Float64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  c.SetFloat64(x/y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) LOGADD(a, b, t Float64) Float64 {
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
func (c Float64) LOGSUB(a, b, t Float64) Float64 {
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
func (c Float64) POW(a, k Float64) Float64 {
  x := a.GetFloat64()
  y := k.GetFloat64()
  c.SetFloat64(math.Pow(x, y))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) SQRT(a Float64) Float64 {
  x := a.GetFloat64()
  c.SetFloat64(math.Sqrt(x))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Float64) EXP(a Float64) Float64 {
  x := a.GetFloat64()
  c.SetFloat64(math.Exp(x))
  return c
}
func (c Float64) LOG(a Float64) Float64 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log(x))
  return c
}
func (c Float64) LOG1P(a Float64) Float64 {
  x := a.GetFloat64()
  c.SetFloat64(math.Log1p(x))
  return c
}
