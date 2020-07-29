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
func (a *Real64) EQUALS(b *Real64, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a *Real64) GREATER(b *Real64) bool {
  return a.GetFloat64() > b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a *Real64) SMALLER(b *Real64) bool {
  return a.GetFloat64() < b.GetFloat64()
}
/* -------------------------------------------------------------------------- */
func (a *Real64) SIGN() int {
  if a.GetFloat64() < float64(0) {
    return -1
  }
  if a.GetFloat64() > float64(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r *Real64) MIN(a, b *Real64) Scalar {
  if a.GetFloat64() < b.GetFloat64() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r *Real64) MAX(a, b *Real64) Scalar {
  if a.GetFloat64() > b.GetFloat64() {
    r.SET(a)
  } else {
    r.SET(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c *Real64) ABS(a *Real64) Scalar {
  if c.Sign() == -1 {
    c.NEG(a)
  } else {
    c.SET(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c *Real64) NEG(a *Real64) *Real64 {
  x := a.GetFloat64()
  return c.realMonadic(a, -x, -1, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real64) ADD(a, b *Real64) *Real64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.realDyadic(a, b, x+y, 1, 1, 0, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real64) SUB(a, b *Real64) *Real64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.realDyadic(a, b, x-y, 1, -1, 0, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real64) MUL(a, b *Real64) *Real64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.realDyadic(a, b, x*y, y, x, 1, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real64) DIV(a, b *Real64) *Real64 {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.realDyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}
/* -------------------------------------------------------------------------- */
func (c *Real64) LOGADD(a, b, t *Real64) *Real64 {
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
func (c *Real64) LOGSUB(a, b, t *Real64) *Real64 {
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
func (c *Real64) POW(a, k *Real64) *Real64 {
  x := a.GetFloat64()
  y := k.GetFloat64()
  v0 := math.Pow(x, y)
  if k.GetOrder() >= 1 {
    f1 := func() (float64, float64) {
      f10 := math.Pow(x, y-1)*y
      f01 := math.Pow(x, y-0)*math.Log(x)
      return f10, f01
    }
    f2 := func() (float64, float64, float64) {
      f11 := math.Pow(x, y-1)*(1 + y*math.Log(x))
      f20 := math.Pow(x, y-2)*(y - 1)*y
      f02 := math.Pow(x, y-0)*math.Log(x)*math.Log(x)
      return f11, f20, f02
    }
    return c.realDyadicLazy(a, k, v0, f1, f2)
  } else {
    f1 := func() (float64) {
      return math.Pow(x, y-1)*y
    }
    f2 := func() (float64) {
      return math.Pow(x, y-2)*(y - 1)*y
    }
    return c.realMonadicLazy(a, v0, f1, f2)
  }
}
/* -------------------------------------------------------------------------- */
func (c *Real64) SQRT(a *Real64) *Real64 {
  x := a.GetFloat64()
  y := 0.5
  v0 := math.Pow(x, y)
  f1 := func() (float64) {
    return math.Pow(x, y-1)*y
  }
  f2 := func() (float64) {
    return math.Pow(x, y-2)*(y - 1)*y
  }
  return c.realMonadicLazy(a, v0, f1, f2)
}
/* -------------------------------------------------------------------------- */
func (c *Real64) EXP(a *Real64) *Real64 {
  x := a.GetFloat64()
  v0 := math.Exp(x)
  f1 := func() float64 { return v0 }
  f2 := func() float64 { return v0 }
  return c.realMonadicLazy(a, v0, f1, f2)
}
func (c *Real64) LOG(a *Real64) *Real64 {
  x := a.GetFloat64()
  v0 := math.Log(x)
  f1 := func() float64 { return 1/x }
  f2 := func() float64 { return -1/(x*x) }
  return c.realMonadicLazy(a, v0, f1, f2)
}
func (c *Real64) LOG1P(a *Real64) *Real64 {
  x := a.GetFloat64()
  v0 := math.Log1p(x)
  f1 := func() float64 { return 1/ (1+x) }
  f2 := func() float64 { return -1/((1+x)*(1+x)) }
  return c.realMonadicLazy(a, v0, f1, f2)
}
