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

const bareRealDebug = false

/* -------------------------------------------------------------------------- */

func checkBare(b ConstScalar) {
  if bareRealDebug {
    if b.GetOrder() > 0 {
      panic("BareReal cannot carry any derivates!")
    }
  }
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Equals(b ConstScalar, epsilon float64) bool {
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Greater(b ConstScalar) bool {
  return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Smaller(b ConstScalar) bool {
  return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) Min(a, b ConstScalar) Scalar {
  if a.GetValue() < b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) Max(a, b ConstScalar) Scalar {
  if a.GetValue() > b.GetValue() {
    r.Set(a)
  } else {
    r.Set(b)
  }
  return r
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

/* -------------------------------------------------------------------------- */

func (c *BareReal) Abs(a ConstScalar) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  } else {
    c.Set(a)
  }
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Neg(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(-a.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Add(a, b ConstScalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() + b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sub(a, b ConstScalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() - b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Mul(a, b ConstScalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() * b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Div(a, b ConstScalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() / b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) LogAdd(a, b ConstScalar, t Scalar) Scalar {
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

func (c *BareReal) LogSub(a, b ConstScalar, t Scalar) Scalar {
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

func (c *BareReal) Log1pExp(a ConstScalar) Scalar {
  v := a.GetValue()
  if v <= -37.0 {
    c.Exp(a)
  } else
  if v <=  18.0 {
    c.Exp(a)
    c.Log1p(c)
  } else
  if v <=  33.3 {
    c.Neg(a)
    c.Exp(a)
    c.Add(c, a)
  } else {
    c.Set(a)
  }
  return c
}

func (c *BareReal) Sigmoid(a ConstScalar, t Scalar) Scalar {
  if a.GetValue() >= 0 {
    c.Neg(a)
    c.Exp(c)
    c.Add(c, ConstReal(1.0))
    c.Div(ConstReal(1.0), c)
  } else {
    t.Exp(a)
    c.Set(t)
    t.Add(t, ConstReal(1.0))
    c.Div(c, t)
  }
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Pow(a, k ConstScalar) Scalar {
  checkBare(a)
  checkBare(k)
  *c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sqrt(a ConstScalar) Scalar {
  checkBare(a)
  return c.Pow(a, ConstReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sin(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sin(a.GetValue()))
  return c
}

func (c *BareReal) Sinh(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sinh(a.GetValue()))
  return c
}

func (c *BareReal) Cos(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cos(a.GetValue()))
  return c
}

func (c *BareReal) Cosh(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cosh(a.GetValue()))
  return c
}

func (c *BareReal) Tan(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tan(a.GetValue()))
  return c
}

func (c *BareReal) Tanh(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tanh(a.GetValue()))
  return c
}

func (c *BareReal) Exp(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Exp(a.GetValue()))
  return c
}

func (c *BareReal) Log(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Log(a.GetValue()))
  return c
}

func (c *BareReal) Log1p(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Log1p(a.GetValue()))
  return c
}

func (c *BareReal) Logistic(a ConstScalar) Scalar {
  checkBare(a)
  c.Neg(a)
  c.Exp(c)
  c.Add(ConstReal(1.0), c)
  c.Div(ConstReal(1.0), c)
  return c
}

func (c *BareReal) Erf(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erf(a.GetValue()))
  return c
}

func (c *BareReal) Erfc(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erfc(a.GetValue()))
  return c
}

func (c *BareReal) LogErfc(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(special.LogErfc(a.GetValue()))
  return c
}

func (c *BareReal) Gamma(a ConstScalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Gamma(a.GetValue()))
  return c
}

func (c *BareReal) Lgamma(a ConstScalar) Scalar {
  checkBare(a)
  v, s := math.Lgamma(a.GetValue())
  if s == -1 {
    v = math.NaN()
  }
  *c = BareReal(v)
  return c
}

func (c *BareReal) Mlgamma(a ConstScalar, k int) Scalar {
  checkBare(a)
  *c = BareReal(special.Mlgamma(a.GetValue(), k))
  return c
}

func (c *BareReal) GammaP(a float64, x ConstScalar) Scalar {
  checkBare(x)
  *c = BareReal(special.GammaP(a, x.GetValue()))
  return c
}

func (c *BareReal) BesselI(v float64, x ConstScalar) Scalar {
  checkBare(x)
  *c = BareReal(special.BesselI(v, x.GetValue()))
  return c
}

func (c *BareReal) LogBesselI(v float64, x ConstScalar) Scalar {
  checkBare(x)
  *c = BareReal(special.LogBesselI(v, x.GetValue()))
  return c
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) SmoothMax(x ConstVector, alpha ConstReal, t [2]Scalar) Scalar {
  r   .Reset()
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

func (r *BareReal) LogSmoothMax(x ConstVector, alpha ConstReal, t [3]Scalar) Scalar {
  r   .Reset()
  t[2].SetValue(math.Inf(-1))
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

func (r *BareReal) Vmean(a ConstVector) Scalar {
  r.Reset()
  for i := 0; i < a.Dim(); i++ {
    r.Add(r, a.ConstAt(i))
  }
  return r.Div(r, NewBareReal(float64(a.Dim())))
}

func (r *BareReal) VdotV(a, b ConstVector) Scalar {
  if a.Dim() != b.Dim() {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullBareReal()
  for i := 0; i < a.Dim(); i++ {
    t.Mul(a.ConstAt(i), b.ConstAt(i))
    r.Add(r, t)
  }
  return r
}

func (r *BareReal) Vnorm(a ConstVector) Scalar {
  r.Reset()
  t := BareReal(0.0)
  for it := a.ConstIterator(); it.Ok(); it.Next() {
    t.Pow(it.GetConst(), ConstReal(2.0))
    r.Add(r, &t)
  }
  r.Sqrt(r)
  return r
}

func (r *BareReal) Mtrace(a ConstMatrix) Scalar {
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
func (r *BareReal) Mnorm(a ConstMatrix) Scalar {
  n, m := a.Dims()
  if n == 0 || m == 0 {
    return nil
  }
  c := NewBareReal(2.0)
  t := NewScalar(r.Type(), 0.0)
  v := a.AsConstVector()
  r.Pow(v.ConstAt(0), NewBareReal(2.0))
  for i := 1; i < v.Dim(); i++ {
    t.Pow(v.ConstAt(i), c)
    r.Add(r, t)
  }
  return r
}
