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

//import "fmt"
import "math"

import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func (a *Real) Equals(b ConstScalar, epsilon float64) bool {
	return math.Abs(a.GetValue()-b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *Real) Greater(b ConstScalar) bool {
	return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *Real) Smaller(b ConstScalar) bool {
	return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *Real) Sign() int {
	if a.GetValue() < 0.0 {
		return -1
	}
	if a.GetValue() > 0.0 {
		return 1
	}
	return 0
}

/* -------------------------------------------------------------------------- */

func (r *Real) Min(a, b ConstScalar) Scalar {
	if a.GetValue() < b.GetValue() {
		r.Set(a)
	} else {
		r.Set(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (r *Real) Max(a, b ConstScalar) Scalar {
	if a.GetValue() > b.GetValue() {
		r.Set(a)
	} else {
		r.Set(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (c *Real) Abs(a ConstScalar) Scalar {
	switch a.Sign() {
	case -1:
		c.Neg(a)
	case 0:
		c.Reset()
	case 1:
		c.Set(a)
	}
	return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Neg(a ConstScalar) Scalar {
	x := a.GetValue()
	return c.monadic(a, -x, -1, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Add(a, b ConstScalar) Scalar {
	x := a.GetValue()
	y := b.GetValue()
	return c.dyadic(a, b, x+y, 1, 1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sub(a, b ConstScalar) Scalar {
	x := a.GetValue()
	y := b.GetValue()
	return c.dyadic(a, b, x-y, 1, -1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Mul(a, b ConstScalar) Scalar {
	x := a.GetValue()
	y := b.GetValue()
	return c.dyadic(a, b, x*y, y, x, 1, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Div(a, b ConstScalar) Scalar {
	x := a.GetValue()
	y := b.GetValue()
	return c.dyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}

/* -------------------------------------------------------------------------- */

func (c *Real) LogAdd(a, b ConstScalar, t Scalar) Scalar {
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

func (c *Real) LogSub(a, b ConstScalar, t Scalar) Scalar {
	if math.IsInf(b.GetValue(), -1) {
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

func (c *Real) Log1pExp(a ConstScalar) Scalar {
	v := a.GetValue()
	if v <= -37.0 {
		c.Exp(a)
	} else if v <= 18.0 {
		c.Exp(a)
		c.Log1p(c)
	} else if v <= 33.3 {
		c.Neg(a)
		c.Exp(a)
		c.Add(c, a)
	} else {
		c.Set(a)
	}
	return c
}

func (c *Real) Sigmoid(a ConstScalar, t Scalar) Scalar {
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

func (c *Real) Pow(a, k ConstScalar) Scalar {
	x := a.GetValue()
	y := k.GetValue()
	v0 := math.Pow(x, y)
	if k.GetOrder() >= 1 {
		f1 := func() (float64, float64) {
			f10 := math.Pow(x, y-1) * y
			f01 := math.Pow(x, y-0) * math.Log(x)
			return f10, f01
		}
		f2 := func() (float64, float64, float64) {
			f11 := math.Pow(x, y-1) * (1 + y*math.Log(x))
			f20 := math.Pow(x, y-2) * (y - 1) * y
			f02 := math.Pow(x, y-0) * math.Log(x) * math.Log(x)
			return f11, f20, f02
		}
		return c.dyadicLazy(a, k, v0, f1, f2)
	} else {
		f1 := func() float64 {
			return math.Pow(x, y-1) * y
		}
		f2 := func() float64 {
			return math.Pow(x, y-2) * (y - 1) * y
		}
		return c.monadicLazy(a, v0, f1, f2)
	}
}

func (c *Real) RealPow(a, k *Real) *Real {
	x := a.GetValue()
	y := k.GetValue()
	v0 := math.Pow(x, y)
	if k.GetOrder() >= 1 {
		f1 := func() (float64, float64) {
			f10 := math.Pow(x, y-1) * y
			f01 := math.Pow(x, y-0) * math.Log(x)
			return f10, f01
		}
		f2 := func() (float64, float64, float64) {
			f11 := math.Pow(x, y-1) * (1 + y*math.Log(x))
			f20 := math.Pow(x, y-2) * (y - 1) * y
			f02 := math.Pow(x, y-0) * math.Log(x) * math.Log(x)
			return f11, f20, f02
		}
		return c.realDyadicLazy(a, k, v0, f1, f2)
	} else {
		f1 := func() float64 {
			return math.Pow(x, y-1) * y
		}
		f2 := func() float64 {
			return math.Pow(x, y-2) * (y - 1) * y
		}
		return c.realMonadicLazy(a, v0, f1, f2)
	}
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sqrt(a ConstScalar) Scalar {
	return c.Pow(a, ConstReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sin(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Sin(x)
	f1 := func() float64 { return math.Cos(x) }
	f2 := func() float64 { return -math.Sin(x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Sinh(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Sinh(x)
	f1 := func() float64 { return math.Cosh(x) }
	f2 := func() float64 { return math.Sinh(x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Cos(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Cos(x)
	f1 := func() float64 { return -math.Sin(x) }
	f2 := func() float64 { return -math.Cos(x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Cosh(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Cosh(x)
	f1 := func() float64 { return math.Sinh(x) }
	f2 := func() float64 { return math.Cosh(x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Tan(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Tan(x)
	f1 := func() float64 { return 1.0 + math.Pow(math.Tan(x), 2) }
	f2 := func() float64 { return 2.0 * math.Tan(x) * f1() }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Tanh(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Tanh(x)
	f1 := func() float64 { return 1.0 - math.Pow(math.Tanh(x), 2) }
	f2 := func() float64 { return -2.0 * math.Tanh(x) * f1() }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Exp(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Exp(x)
	f1 := func() float64 { return v0 }
	f2 := func() float64 { return v0 }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Log(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Log(x)
	f1 := func() float64 { return 1 / x }
	f2 := func() float64 { return -1 / (x * x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Log1p(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Log1p(x)
	f1 := func() float64 { return 1 / (1 + x) }
	f2 := func() float64 { return -1 / ((1 + x) * (1 + x)) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Logistic(a ConstScalar) Scalar {
	c.Neg(a)
	c.Exp(c)
	c.Add(ConstReal(1.0), c)
	c.Div(ConstReal(1.0), c)
	return c
}

func (c *Real) Erf(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Erf(x)
	f1 := func() float64 {
		return 2.0 / (math.Exp(x*x) * special.M_SQRTPI)
	}
	f2 := func() float64 {
		return -4.0 / (math.Exp(x*x) * special.M_SQRTPI) * x
	}
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Erfc(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Erf(x)
	f1 := func() float64 {
		return -2.0 / (math.Exp(x*x) * special.M_SQRTPI)
	}
	f2 := func() float64 {
		return 4.0 / (math.Exp(x*x) * special.M_SQRTPI) * x
	}
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) LogErfc(a ConstScalar) Scalar {
	x := a.GetValue()
	t := math.Erfc(x)
	v0 := special.LogErfc(x)
	f1 := func() float64 {
		return -2.0 / (math.Exp(a.GetValue()*a.GetValue()) * special.M_SQRTPI * t)
	}
	f2 := func() float64 {
		return 4.0 * (math.Exp(x*x)*special.M_SQRTPI*t*x - 1) / (math.Exp(2*x*x) * math.Pi * t * t)
	}
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Gamma(a ConstScalar) Scalar {
	x := a.GetValue()
	v0 := math.Gamma(x)
	f1 := func() float64 {
		v1 := special.Digamma(x)
		return v0 * v1
	}
	f2 := func() float64 {
		v1 := special.Digamma(x)
		v2 := special.Trigamma(x)
		return v0 * (v1*v1 + v2)
	}
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Lgamma(a ConstScalar) Scalar {
	x := a.GetValue()
	v0, s := math.Lgamma(a.GetValue())
	if s == -1 {
		v0 = math.NaN()
	}
	f1 := func() float64 { return special.Digamma(x) }
	f2 := func() float64 { return special.Trigamma(x) }
	return c.monadicLazy(a, v0, f1, f2)
}

func (c *Real) Mlgamma(a ConstScalar, k int) Scalar {
	x := a.GetValue()
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

func (c *Real) GammaP(a float64, b ConstScalar) Scalar {
	x := b.GetValue()
	v0 := special.GammaP(a, x)
	f1 := func() float64 {
		return special.GammaPfirstDerivative(a, x)
	}
	f2 := func() float64 {
		return special.GammaPsecondDerivative(a, x)
	}
	return c.monadicLazy(b, v0, f1, f2)
}

func (c *Real) BesselI(v float64, b ConstScalar) Scalar {
	x := b.GetValue()
	v0 := special.BesselI(v, x)
	f1 := func() float64 {
		v1 := special.BesselI(v-1.0, x)
		return v1 - v/x*v0
	}
	f2 := func() float64 {
		v1 := special.BesselI(v-2.0, x)
		v2 := special.BesselI(v+2.0, x)
		return 0.25 * (v1 + 2.0*v0 + v2)
	}
	return c.monadicLazy(b, v0, f1, f2)
}

func (c *Real) LogBesselI(v float64, b ConstScalar) Scalar {
	x := b.GetValue()
	v0 := special.LogBesselI(v, x)
	f1 := func() float64 {
		v1 := special.LogBesselI(v-1.0, x)
		return math.Exp(v1-v0) - v/x
	}
	f2 := func() float64 {
		v1 := special.LogBesselI(v-1.0, x)
		v2 := special.LogBesselI(v-2.0, x)
		v3 := special.LogBesselI(v+2.0, x)
		t1 := 0.25 * (math.Exp(v2-v0) + 2.0 + math.Exp(v3-v0))
		t2 := math.Exp(v1-v0) - v/x
		return t1 - t2*t2
	}
	return c.monadicLazy(b, v0, f1, f2)
}

/* -------------------------------------------------------------------------- */

func (r *Real) SmoothMax(x ConstVector, alpha ConstReal, t [2]Scalar) Scalar {
	r.Reset()
	t[1].Reset()
	for i := 0; i < x.Dim(); i++ {
		t[0].Mul(alpha, x.ConstAt(i))
		t[0].Exp(t[0])
		t[1].Add(t[1], t[0])
		t[0].Mul(t[0], x.ConstAt(i))
		r.Add(r, t[0])
	}
	r.Div(r, t[1])
	return r
}

func (r *Real) LogSmoothMax(x ConstVector, alpha ConstReal, t [3]Scalar) Scalar {
	r.Reset()
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

func (r *Real) Vmean(a ConstVector) Scalar {
	r.Reset()
	for i := 0; i < a.Dim(); i++ {
		r.Add(r, a.ConstAt(i))
	}
	return r.Div(r, ConstReal(float64(a.Dim())))
}

func (r *Real) VdotV(a, b ConstVector) Scalar {
	if a.Dim() != b.Dim() {
		panic("vector dimensions do not match")
	}
	r.Reset()
	t := NullReal()
	for i := 0; i < a.Dim(); i++ {
		t.Mul(a.ConstAt(i), b.ConstAt(i))
		r.Add(r, t)
	}
	return r
}

func (r *Real) Vnorm(a ConstVector) Scalar {
	r.Reset()
	t := NewReal(0.0)
	for it := a.ConstIterator(); it.Ok(); it.Next() {
		t.Pow(it.GetConst(), ConstReal(2.0))
		r.Add(r, t)
	}
	r.Sqrt(r)
	return r
}

func (r *Real) Mtrace(a ConstMatrix) Scalar {
	n, m := a.Dims()
	if n != m {
		panic("not a square matrix")
	}
	if n == 0 {
		return nil
	}
	r.Reset()
	for i := 0; i < n; i++ {
		r.Add(r, a.ConstAt(i, i))
	}
	return r
}

// Frobenius norm.
func (r *Real) Mnorm(a ConstMatrix) Scalar {
	n, m := a.Dims()
	if n == 0 || m == 0 {
		return nil
	}
	c := ConstReal(2.0)
	t := NewScalar(r.Type(), 0.0)
	v := a.AsConstVector()
	r.Pow(v.ConstAt(0), ConstReal(2.0))
	for i := 1; i < v.Dim(); i++ {
		t.Pow(v.ConstAt(i), c)
		r.Add(r, t)
	}
	return r
}
