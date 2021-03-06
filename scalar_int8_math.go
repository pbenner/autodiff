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
import "github.com/pbenner/autodiff/special"
/* -------------------------------------------------------------------------- */
func (a Int8) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a Int8) Greater(b ConstScalar) bool {
  return a.GetInt8() > b.GetInt8()
}
/* -------------------------------------------------------------------------- */
func (a Int8) Smaller(b ConstScalar) bool {
  return a.GetInt8() < b.GetInt8()
}
/* -------------------------------------------------------------------------- */
func (a Int8) Sign() int {
  if a.GetInt8() < int8(0) {
    return -1
  }
  if a.GetInt8() > int8(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r Int8) Min(a, b ConstScalar) Scalar {
  if a.GetInt8() < b.GetInt8() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r Int8) Max(a, b ConstScalar) Scalar {
  if a.GetInt8() > b.GetInt8() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c Int8) Abs(a ConstScalar) Scalar {
  switch a.Sign() {
  case -1: c.Neg(a)
  case 0: c.Reset()
  case 1: c.Set(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Neg(a ConstScalar) Scalar {
  x := a.GetInt8()
  c.SetInt8(-x)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Add(a, b ConstScalar) Scalar {
  x := a.GetInt8()
  y := b.GetInt8()
  c.SetInt8(x+y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Sub(a, b ConstScalar) Scalar {
  x := a.GetInt8()
  y := b.GetInt8()
  c.SetInt8(x-y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Mul(a, b ConstScalar) Scalar {
  x := a.GetInt8()
  y := b.GetInt8()
  c.SetInt8(x*y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Div(a, b ConstScalar) Scalar {
  x := a.GetInt8()
  y := b.GetInt8()
  c.SetInt8(x/y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) LogAdd(a, b ConstScalar, t Scalar) Scalar {
  if a.Greater(b) {
    // swap
    a, b = b, a
  }
  if math.IsInf(a.GetFloat64(), 0) {
    // cases:
    //  i) a = -Inf and b >= a    => c = b
    // ii) a =  Inf and b  = Inf  => c = Inf
    c.Set(b)
    return c
  }
  t.Sub(a, b)
  t.Exp(t)
  t.Log1p(t)
  c.Add(t, b)
  return c
}
func (c Int8) LogSub(a, b ConstScalar, t Scalar) Scalar {
  if math.IsInf(b.GetFloat64(), -1) {
    c.Set(a)
    return c
  }
  //   log(exp(a) - exp(b))
  // = log(1 - exp(b-a)) + a
  t.Sub(b, a)
  t.Exp(t)
  t.Neg(t)
  t.Log1p(t)
  c.Add(t, a)
  return c
}
func (c Int8) Log1pExp(a ConstScalar) Scalar {
  v := a.GetFloat64()
  if v <= -37.0 {
    c.Exp(a)
  } else
  if v <= 18.0 {
    c.Exp(a)
    c.Log1p(c)
  } else
  if v <= 33.3 {
    c.Neg(a)
    c.Exp(a)
    c.Add(c, a)
  } else {
    c.Set(a)
  }
  return c
}
func (c Int8) Sigmoid(a ConstScalar, t Scalar) Scalar {
  if a.GetFloat64() >= 0 {
    c.Neg(a)
    c.Exp(c)
    c.Add(c, ConstInt8(1.0))
    c.Div(ConstInt8(1.0), c)
  } else {
    t.Exp(a)
    c.Set(t)
    t.Add(t, ConstInt8(1.0))
    c.Div(c, t)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Pow(a, k ConstScalar) Scalar {
  x := a.GetFloat64()
  y := k.GetFloat64()
  c.SetFloat64(math.Pow(x, y))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int8) Sqrt(a ConstScalar) Scalar {
  return c.Pow(a, ConstFloat64(0.5))
}
/* -------------------------------------------------------------------------- */
func (c Int8) Sin(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Sin(x))
  return c
}
func (c Int8) Sinh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Sinh(x))
  return c
}
func (c Int8) Cos(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Cos(x))
  return c
}
func (c Int8) Cosh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Cosh(x))
  return c
}
func (c Int8) Tan(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Tan(x))
  return c
}
func (c Int8) Tanh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Tanh(x))
  return c
}
func (c Int8) Exp(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Exp(x))
  return c
}
func (c Int8) Log(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Log(x))
  return c
}
func (c Int8) Log1p(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Log1p(x))
  return c
}
func (c Int8) Logistic(a ConstScalar) Scalar {
  c.Neg(a)
  c.Exp(c)
  c.Add(ConstInt8(1.0), c)
  c.Div(ConstInt8(1.0), c)
  return c
}
func (c Int8) Erf(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Erf(x))
  return c
}
func (c Int8) Erfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Erf(x))
  return c
}
func (c Int8) LogErfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(special.LogErfc(x))
  return c
}
func (c Int8) Gamma(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Gamma(x))
  return c
}
func (c Int8) Lgamma(a ConstScalar) Scalar {
  v0, s := math.Lgamma(a.GetFloat64())
  if s == -1 {
    v0 = math.NaN()
  }
  c.SetFloat64(v0)
  return c
}
func (c Int8) Mlgamma(a ConstScalar, k int) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(special.Mlgamma(x, k))
  return c
}
func (c Int8) GammaP(a float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.GammaP(a, x))
  return c
}
func (c Int8) BesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.BesselI(v, x))
  return c
}
func (c Int8) LogBesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.LogBesselI(v, x))
  return c
}
/* -------------------------------------------------------------------------- */
func (r Int8) SmoothMax(x ConstVector, alpha ConstFloat64, t [2]Scalar) Scalar {
  r .Reset()
  t[1].Reset()
  for i := 0; i < x.Dim(); i++ {
    t[0].Mul(alpha, x.ConstAt(i))
    t[0].Exp(t[0])
    t[1].Add(t[1], t[0])
    t[0].Mul(t[0], x.ConstAt(i))
    r .Add(r , t[0])
  }
  r.Div(r, t[1])
  return r
}
func (r Int8) LogSmoothMax(x ConstVector, alpha ConstFloat64, t [3]Scalar) Scalar {
  r .Reset()
  t[2].SetFloat64(math.Inf(-1))
  for i := 0; i < x.Dim(); i++ {
    t[0].Mul(x.ConstAt(i), alpha)
    t[2].LogAdd(t[2], t[0], t[1])
    t[1].Log(x.ConstAt(i))
    t[0].Add(t[0], t[1])
    r.LogAdd(r, t[0], t[1])
  }
  r.Sub(r, t[2])
  r.Exp(r)
  return r
}
func (r Int8) Vmean(a ConstVector) Scalar {
  r.Reset()
  for i := 0; i < a.Dim(); i++ {
    r.Add(r, a.ConstAt(i))
  }
  return r.Div(r, ConstInt8(float64(a.Dim())))
}
func (r Int8) VdotV(a, b ConstVector) Scalar {
  if a.Dim() != b.Dim() {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullInt8()
  for i := 0; i < a.Dim(); i++ {
    t.Mul(a.ConstAt(i), b.ConstAt(i))
    r.Add(r, t)
  }
  return r
}
func (r Int8) Vnorm(a ConstVector) Scalar {
  r.Reset()
  t := NullInt8()
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    t.Pow(it.GetConst(), ConstInt8(2.0))
    r.Add(r, t)
  }
  r.Sqrt(r)
  return r
}
func (r Int8) Mtrace(a ConstMatrix) Scalar {
  n, m := a.Dims()
  if n != m {
    panic("not a square matrix")
  }
  if n == 0 {
    return nil
  }
  r.Reset()
  for i := 0; i < n; i++ {
    r.Add(r, a.ConstAt(i,i))
  }
  return r
}
// Frobenius norm.
func (r Int8) Mnorm(a ConstMatrix) Scalar {
  n, m := a.Dims()
  if n == 0 || m == 0 {
    return nil
  }
  t := NewScalar(r.Type(), 0.0)
  v := a.AsConstVector()
  r.Pow(v.ConstAt(0), ConstInt8(2.0))
  for i := 1; i < v.Dim(); i++ {
    t.Pow(v.ConstAt(i), ConstInt8(2.0))
    r.Add(r, t)
  }
  return r
}
