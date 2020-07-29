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
func (a Int) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a Int) Greater(b ConstScalar) bool {
  return a.GetInt() > b.GetInt()
}
/* -------------------------------------------------------------------------- */
func (a Int) Smaller(b ConstScalar) bool {
  return a.GetInt() < b.GetInt()
}
/* -------------------------------------------------------------------------- */
func (a Int) Sign() int {
  if a.GetInt() < int(0) {
    return -1
  }
  if a.GetInt() > int(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r Int) Min(a, b ConstScalar) Scalar {
  if a.GetInt() < b.GetInt() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r Int) Max(a, b ConstScalar) Scalar {
  if a.GetInt() > b.GetInt() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c Int) Abs(a ConstScalar) Scalar {
  switch a.Sign() {
  case -1: c.Neg(a)
  case 0: c.Reset()
  case 1: c.Set(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Neg(a ConstScalar) Scalar {
  x := a.GetInt()
  c.SetInt(-x)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Add(a, b ConstScalar) Scalar {
  x := a.GetInt()
  y := b.GetInt()
  c.SetInt(x+y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Sub(a, b ConstScalar) Scalar {
  x := a.GetInt()
  y := b.GetInt()
  c.SetInt(x-y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Mul(a, b ConstScalar) Scalar {
  x := a.GetInt()
  y := b.GetInt()
  c.SetInt(x*y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Div(a, b ConstScalar) Scalar {
  x := a.GetInt()
  y := b.GetInt()
  c.SetInt(x/y)
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) LogAdd(a, b ConstScalar, t Scalar) Scalar {
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
func (c Int) LogSub(a, b ConstScalar, t Scalar) Scalar {
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
func (c Int) Log1pExp(a ConstScalar) Scalar {
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
func (c Int) Sigmoid(a ConstScalar, t Scalar) Scalar {
  if a.GetFloat64() >= 0 {
    c.Neg(a)
    c.Exp(c)
    c.Add(c, ConstInt(1.0))
    c.Div(ConstInt(1.0), c)
  } else {
    t.Exp(a)
    c.Set(t)
    t.Add(t, ConstInt(1.0))
    c.Div(c, t)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Pow(a, k ConstScalar) Scalar {
  x := a.GetFloat64()
  y := k.GetFloat64()
  c.SetFloat64(math.Pow(x, y))
  return c
}
/* -------------------------------------------------------------------------- */
func (c Int) Sqrt(a ConstScalar) Scalar {
  return c.Pow(a, ConstFloat64(0.5))
}
/* -------------------------------------------------------------------------- */
func (c Int) Sin(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Sin(x))
  return c
}
func (c Int) Sinh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Sinh(x))
  return c
}
func (c Int) Cos(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Cos(x))
  return c
}
func (c Int) Cosh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Cosh(x))
  return c
}
func (c Int) Tan(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Tan(x))
  return c
}
func (c Int) Tanh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Tanh(x))
  return c
}
func (c Int) Exp(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Exp(x))
  return c
}
func (c Int) Log(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Log(x))
  return c
}
func (c Int) Log1p(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Log1p(x))
  return c
}
func (c Int) Logistic(a ConstScalar) Scalar {
  c.Neg(a)
  c.Exp(c)
  c.Add(ConstInt(1.0), c)
  c.Div(ConstInt(1.0), c)
  return c
}
func (c Int) Erf(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Erf(x))
  return c
}
func (c Int) Erfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Erf(x))
  return c
}
func (c Int) LogErfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(special.LogErfc(x))
  return c
}
func (c Int) Gamma(a ConstScalar) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(math.Gamma(x))
  return c
}
func (c Int) Lgamma(a ConstScalar) Scalar {
  v0, s := math.Lgamma(a.GetFloat64())
  if s == -1 {
    v0 = math.NaN()
  }
  c.SetFloat64(v0)
  return c
}
func (c Int) Mlgamma(a ConstScalar, k int) Scalar {
  x := a.GetFloat64()
  c.SetFloat64(special.Mlgamma(x, k))
  return c
}
func (c Int) GammaP(a float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.GammaP(a, x))
  return c
}
func (c Int) BesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.BesselI(v, x))
  return c
}
func (c Int) LogBesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  c.SetFloat64(special.LogBesselI(v, x))
  return c
}
/* -------------------------------------------------------------------------- */
func (r Int) SmoothMax(x ConstVector, alpha ConstFloat64, t [2]Scalar) Scalar {
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
func (r Int) LogSmoothMax(x ConstVector, alpha ConstFloat64, t [3]Scalar) Scalar {
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
func (r Int) Vmean(a ConstVector) Scalar {
  r.Reset()
  for i := 0; i < a.Dim(); i++ {
    r.Add(r, a.ConstAt(i))
  }
  return r.Div(r, ConstInt(float64(a.Dim())))
}
func (r Int) VdotV(a, b ConstVector) Scalar {
  if a.Dim() != b.Dim() {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullInt()
  for i := 0; i < a.Dim(); i++ {
    t.Mul(a.ConstAt(i), b.ConstAt(i))
    r.Add(r, t)
  }
  return r
}
func (r Int) Vnorm(a ConstVector) Scalar {
  r.Reset()
  t := NullInt()
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    t.Pow(it.GetConst(), ConstInt(2.0))
    r.Add(r, t)
  }
  r.Sqrt(r)
  return r
}
func (r Int) Mtrace(a ConstMatrix) Scalar {
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
func (r Int) Mnorm(a ConstMatrix) Scalar {
  n, m := a.Dims()
  if n == 0 || m == 0 {
    return nil
  }
  t := NewScalar(r.Type(), 0.0)
  v := a.AsConstVector()
  r.Pow(v.ConstAt(0), ConstInt(2.0))
  for i := 1; i < v.Dim(); i++ {
    t.Pow(v.ConstAt(i), ConstInt(2.0))
    r.Add(r, t)
  }
  return r
}
