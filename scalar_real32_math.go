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
func (a *Real32) Equals(b ConstScalar, epsilon float64) bool {
  v1 := a.GetFloat64()
  v2 := b.GetFloat64()
  return math.Abs(v1 - v2) < epsilon ||
        (math.IsNaN(v1) && math.IsNaN(v2)) ||
        (math.IsInf(v1, 1) && math.IsInf(v2, 1)) ||
        (math.IsInf(v1, -1) && math.IsInf(v2, -1))
}
/* -------------------------------------------------------------------------- */
func (a *Real32) Greater(b ConstScalar) bool {
  return a.GetFloat32() > b.GetFloat32()
}
/* -------------------------------------------------------------------------- */
func (a *Real32) Smaller(b ConstScalar) bool {
  return a.GetFloat32() < b.GetFloat32()
}
/* -------------------------------------------------------------------------- */
func (a *Real32) Sign() int {
  if a.GetFloat32() < float32(0) {
    return -1
  }
  if a.GetFloat32() > float32(0) {
    return 1
  }
  return 0
}
/* -------------------------------------------------------------------------- */
func (r *Real32) Min(a, b ConstScalar) Scalar {
  if a.GetFloat32() < b.GetFloat32() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (r *Real32) Max(a, b ConstScalar) Scalar {
  if a.GetFloat32() > b.GetFloat32() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Abs(a ConstScalar) Scalar {
  switch a.Sign() {
  case -1: c.Neg(a)
  case 0: c.Reset()
  case 1: c.Set(a)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Neg(a ConstScalar) Scalar {
  x := a.GetFloat64()
  return c.monadic(a, -x, -1, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Add(a, b ConstScalar) Scalar {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.dyadic(a, b, x+y, 1, 1, 0, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Sub(a, b ConstScalar) Scalar {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.dyadic(a, b, x-y, 1, -1, 0, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Mul(a, b ConstScalar) Scalar {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.dyadic(a, b, x*y, y, x, 1, 0, 0)
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Div(a, b ConstScalar) Scalar {
  x := a.GetFloat64()
  y := b.GetFloat64()
  return c.dyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}
/* -------------------------------------------------------------------------- */
func (c *Real32) LogAdd(a, b ConstScalar, t Scalar) Scalar {
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
func (c *Real32) LogSub(a, b ConstScalar, t Scalar) Scalar {
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
func (c *Real32) Log1pExp(a ConstScalar) Scalar {
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
func (c *Real32) Sigmoid(a ConstScalar, t Scalar) Scalar {
  if a.GetFloat64() >= 0 {
    c.Neg(a)
    c.Exp(c)
    c.Add(c, ConstFloat32(1.0))
    c.Div(ConstFloat32(1.0), c)
  } else {
    t.Exp(a)
    c.Set(t)
    t.Add(t, ConstFloat32(1.0))
    c.Div(c, t)
  }
  return c
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Pow(a, k ConstScalar) Scalar {
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
    return c.dyadicLazy(a, k, v0, f1, f2)
  } else {
    f1 := func() (float64) {
      return math.Pow(x, y-1)*y
    }
    f2 := func() (float64) {
      return math.Pow(x, y-2)*(y - 1)*y
    }
    return c.monadicLazy(a, v0, f1, f2)
  }
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Sqrt(a ConstScalar) Scalar {
  return c.Pow(a, ConstFloat64(0.5))
}
/* -------------------------------------------------------------------------- */
func (c *Real32) Sin(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Sin(x)
  f1 := func() float64 { return math.Cos(x) }
  f2 := func() float64 { return -math.Sin(x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Sinh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Sinh(x)
  f1 := func() float64 { return math.Cosh(x) }
  f2 := func() float64 { return math.Sinh(x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Cos(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Cos(x)
  f1 := func() float64 { return -math.Sin(x) }
  f2 := func() float64 { return -math.Cos(x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Cosh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Cosh(x)
  f1 := func() float64 { return math.Sinh(x) }
  f2 := func() float64 { return math.Cosh(x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Tan(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Tan(x)
  f1 := func() float64 { return 1.0+math.Pow(math.Tan(x), 2) }
  f2 := func() float64 { return 2.0*math.Tan(x)*f1() }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Tanh(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Tanh(x)
  f1 := func() float64 { return 1.0-math.Pow(math.Tanh(x), 2) }
  f2 := func() float64 { return -2.0*math.Tanh(x)*f1() }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Exp(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Exp(x)
  f1 := func() float64 { return v0 }
  f2 := func() float64 { return v0 }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Log(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Log(x)
  f1 := func() float64 { return 1/x }
  f2 := func() float64 { return -1/(x*x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Log1p(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Log1p(x)
  f1 := func() float64 { return 1/ (1+x) }
  f2 := func() float64 { return -1/((1+x)*(1+x)) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Logistic(a ConstScalar) Scalar {
  c.Neg(a)
  c.Exp(c)
  c.Add(ConstFloat32(1.0), c)
  c.Div(ConstFloat32(1.0), c)
  return c
}
func (c *Real32) Erf(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Erf(x)
  f1 := func() float64 {
    return 2.0/(math.Exp(x*x)*special.M_SQRTPI)
  }
  f2 := func() float64 {
    return -4.0/(math.Exp(x*x)*special.M_SQRTPI)*x
  }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Erfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Erf(x)
  f1 := func() float64 {
    return -2.0/(math.Exp(x*x)*special.M_SQRTPI)
  }
  f2 := func() float64 {
    return 4.0/(math.Exp(x*x)*special.M_SQRTPI)*x
  }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) LogErfc(a ConstScalar) Scalar {
  x := a.GetFloat64()
  t := math.Erfc(x)
  v0 := special.LogErfc(x)
  f1 := func() float64 {
    return -2.0/(math.Exp(a.GetFloat64()*a.GetFloat64())*special.M_SQRTPI*t)
  }
  f2 := func() float64 {
    return 4.0*(math.Exp(x*x)*special.M_SQRTPI*t*x - 1)/(math.Exp(2*x*x)*math.Pi*t*t)
  }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Gamma(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0 := math.Gamma(x)
  f1 := func() float64 {
    v1 := special.Digamma(x)
    return v0*v1
  }
  f2 := func() float64 {
    v1 := special.Digamma(x)
    v2 := special.Trigamma(x)
    return v0*(v1*v1 + v2)
  }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Lgamma(a ConstScalar) Scalar {
  x := a.GetFloat64()
  v0, s := math.Lgamma(a.GetFloat64())
  if s == -1 {
    v0 = math.NaN()
  }
  f1 := func() float64 { return special.Digamma(x) }
  f2 := func() float64 { return special.Trigamma(x) }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) Mlgamma(a ConstScalar, k int) Scalar {
  x := a.GetFloat64()
  v0 := special.Mlgamma(x, k)
  f1 := func() float64 {
    s := 0.0
    for j := 1; j <= k; j++ {
      s += special.Digamma(x + float64(1-j)/2.0)
    }
    return s
  }
  f2 := func() float64 {
    s := 0.0
    for j := 1; j <= k; j++ {
      s += special.Trigamma(x + float64(1-j)/2.0)
    }
    return s
  }
  return c.monadicLazy(a, v0, f1, f2)
}
func (c *Real32) GammaP(a float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  v0 := special.GammaP(a, x)
  f1 := func() float64 {
    return special.GammaPfirstDerivative(a, x)
  }
  f2 := func() float64 {
    return special.GammaPsecondDerivative(a, x)
  }
  return c.monadicLazy(b, v0, f1, f2)
}
func (c *Real32) BesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  v0 := special.BesselI(v, x)
  f1 := func() float64 {
    v1 := special.BesselI(v-1.0, x)
    return v1 - v/x*v0
  }
  f2 := func() float64 {
    v1 := special.BesselI(v-2.0, x)
    v2 := special.BesselI(v+2.0, x)
    return 0.25*(v1 + 2.0*v0 + v2)
  }
  return c.monadicLazy(b, v0, f1, f2)
}
func (c *Real32) LogBesselI(v float64, b ConstScalar) Scalar {
  x := b.GetFloat64()
  v0 := special.LogBesselI(v, x)
  f1 := func() float64 {
    v1 := special.LogBesselI(v-1.0, x)
    return math.Exp(v1-v0) - v/x
  }
  f2 := func() float64 {
    v1 := special.LogBesselI(v-1.0, x)
    v2 := special.LogBesselI(v-2.0, x)
    v3 := special.LogBesselI(v+2.0, x)
    t1 := 0.25*(math.Exp(v2-v0) + 2.0 + math.Exp(v3-v0))
    t2 := math.Exp(v1-v0) - v/x
    return t1 - t2*t2
  }
  return c.monadicLazy(b, v0, f1, f2)
}
/* -------------------------------------------------------------------------- */
func (r *Real32) SmoothMax(x ConstVector, alpha ConstFloat64, t [2]Scalar) Scalar {
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
func (r *Real32) LogSmoothMax(x ConstVector, alpha ConstFloat64, t [3]Scalar) Scalar {
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
func (r *Real32) Vmean(a ConstVector) Scalar {
  r.Reset()
  for i := 0; i < a.Dim(); i++ {
    r.Add(r, a.ConstAt(i))
  }
  return r.Div(r, ConstFloat32(float64(a.Dim())))
}
func (r *Real32) VdotV(a, b ConstVector) Scalar {
  if a.Dim() != b.Dim() {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullReal32()
  for i := 0; i < a.Dim(); i++ {
    t.Mul(a.ConstAt(i), b.ConstAt(i))
    r.Add(r, t)
  }
  return r
}
func (r *Real32) Vnorm(a ConstVector) Scalar {
  r.Reset()
  t := NullReal32()
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    t.Pow(it.GetConst(), ConstFloat32(2.0))
    r.Add(r, t)
  }
  r.Sqrt(r)
  return r
}
func (r *Real32) Mtrace(a ConstMatrix) Scalar {
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
func (r *Real32) Mnorm(a ConstMatrix) Scalar {
  n, m := a.Dims()
  if n == 0 || m == 0 {
    return nil
  }
  t := NewScalar(r.Type(), 0.0)
  v := a.AsConstVector()
  r.Pow(v.ConstAt(0), ConstFloat32(2.0))
  for i := 1; i < v.Dim(); i++ {
    t.Pow(v.ConstAt(i), ConstFloat32(2.0))
    r.Add(r, t)
  }
  return r
}
