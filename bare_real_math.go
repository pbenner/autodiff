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

import "math"

import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func checkBare(b Scalar) {
  if b.GetOrder() > 0 {
    panic("BareReal cannot carry any derivates!")
  }
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

func (a *BareReal) BareRealEquals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Greater(b Scalar) bool {
  return a.GetValue() > b.GetValue()
}

func (a *BareReal) BareRealGreater(b *BareReal) bool {
  return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Smaller(b Scalar) bool {
  return a.GetValue() < b.GetValue()
}

func (a *BareReal) BareRealSmaller(b *BareReal) bool {
  return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) Min(a, b Scalar) Scalar {
  if a.GetValue() < b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}

func (r *BareReal) BareRealMin(a, b *BareReal) Scalar {
  if a.GetValue() < b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) Max(a, b Scalar) Scalar {
  if a.GetValue() > b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}

func (r *BareReal) BareRealMax(a, b *BareReal) Scalar {
  if a.GetValue() > b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Abs(a Scalar) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  } else {
    c.Set(a)
  }
  return c
}

func (c *BareReal) BareRealAbs(a *BareReal) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  } else {
    c.Set(a)
  }
  return c
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Sign() int {
  if a.GetValue() < 0.0 {
    return -1
  }
  if a.GetValue() > 0.0 {
    return  1
  }
  return 0
}

func (a *BareReal) RealSign() int {
  if a.GetValue() < 0.0 {
    return -1
  }
  if a.GetValue() > 0.0 {
    return  1
  }
  return 0
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Neg(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(-a.GetValue())
  return c
}

func (c *BareReal) BareRealNeg(a *BareReal) *BareReal {
  *c = BareReal(-a.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Add(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() + b.GetValue())
  return c
}

func (c *BareReal) BareRealAdd(a, b *BareReal) *BareReal {
  *c = *a + *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sub(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() - b.GetValue())
  return c
}

func (c *BareReal) BareRealSub(a, b *BareReal) *BareReal {
  *c = *a - *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Mul(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() * b.GetValue())
  return c
}

func (c *BareReal) BareRealMul(a, b *BareReal) *BareReal {
  *c = *a * *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Div(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() / b.GetValue())
  return c
}

func (c *BareReal) BareRealDiv(a, b *BareReal) *BareReal {
  *c = *a / *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) LogAdd(a, b, t Scalar) Scalar {
  if a.Greater(b) {
    // swap
    a, b = b, a
  }
  if math.IsInf(a.GetValue(), 0) {
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

func (c *BareReal) LogSub(a, b, t Scalar) Scalar {
  if math.IsInf(b.GetValue(), -1) {
    c.Set(a)
    return c
  }
  t.Sub(b, a)
  t.Exp(t)
  t.Neg(t)
  t.Log1p(t)
  c.Add(t, a)
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Pow(a, k Scalar) Scalar {
  checkBare(a)
  checkBare(k)
  *c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

func (c *BareReal) BareRealPow(a, k *BareReal) *BareReal {
  *c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sqrt(a Scalar) Scalar {
  checkBare(a)
  return c.Pow(a, NewBareReal(1.0/2.0))
}

func (c *BareReal) BareRealSqrt(a *BareReal) *BareReal {
  return c.BareRealPow(a, NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sin(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sin(a.GetValue()))
  return c
}

func (c *BareReal) Sinh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sinh(a.GetValue()))
  return c
}

func (c *BareReal) Cos(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cos(a.GetValue()))
  return c
}

func (c *BareReal) Cosh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cosh(a.GetValue()))
  return c
}

func (c *BareReal) Tan(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tan(a.GetValue()))
  return c
}

func (c *BareReal) Tanh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tanh(a.GetValue()))
  return c
}

func (c *BareReal) Exp(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Exp(a.GetValue()))
  return c
}

func (c *BareReal) Log(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Log(a.GetValue()))
  return c
}

func (c *BareReal) Log1p(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Log1p(a.GetValue()))
  return c
}

func (c *BareReal) Erf(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erf(a.GetValue()))
  return c
}

func (c *BareReal) Erfc(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erfc(a.GetValue()))
  return c
}

func (c *BareReal) LogErfc(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(special.LogErfc(a.GetValue()))
  return c
}

func (c *BareReal) Gamma(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Gamma(a.GetValue()))
  return c
}

func (c *BareReal) Lgamma(a Scalar) Scalar {
  checkBare(a)
  v, s := math.Lgamma(a.GetValue())
  if s == -1 {
    v = math.NaN()
  }
  *c = BareReal(v)
  return c
}

func (c *BareReal) Mlgamma(a Scalar, k int) Scalar {
  checkBare(a)
  *c = BareReal(special.Mlgamma(a.GetValue(), k))
  return c
}

func (c *BareReal) GammaP(a float64, x Scalar) Scalar {
  checkBare(x)
  *c = BareReal(special.GammaP(a, x.GetValue()))
  return c
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) Vmean(a Vector) Scalar {
  r.Reset()
  for i := 0; i < a.Dim(); i++ {
    r.Add(r, a.At(i))
  }
  return r.Div(r, NewBareReal(float64(a.Dim())))
}

func (r *BareReal) VdotV(a, b Vector) Scalar {
  if a.Dim() != b.Dim() {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullBareReal()
  for i := 0; i < a.Dim(); i++ {
    t.Mul(a.At(i), b.At(i))
    r.Add(r, t)
  }
  return r
}

func (r *BareReal) Vnorm(a Vector) Scalar {
  r.Reset()
  c := NewBareReal(2.0)
  t := NullScalar(a.ElementType())
  for i := 0; i < a.Dim(); i++ {
    t.Pow(a.At(i), c)
    r.Add(r, t)
  }
  r.Sqrt(r)
  return r
}

func (r *BareReal) Mtrace(a Matrix) Scalar {
  n, m := a.Dims()
  if n != m {
    panic("not a square matrix")
  }
  if n == 0 {
    return nil
  }
  r.Reset()
  for i := 0; i < n; i++ {
    r.Add(r, a.At(i,i))
  }
  return r
}
