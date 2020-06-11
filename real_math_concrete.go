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

//import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func (a *Real) EQUALS(b *Real, epsilon float64) bool {
	return math.Abs(a.GetValue()-b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *Real) GREATER(b *Real) bool {
	return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *Real) SMALLER(b *Real) bool {
	return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *Real) SIGN() int {
	if a.GetValue() < 0.0 {
		return -1
	}
	if a.GetValue() > 0.0 {
		return 1
	}
	return 0
}

/* -------------------------------------------------------------------------- */

func (r *Real) MIN(a, b *Real) Scalar {
	if a.GetValue() < b.GetValue() {
		r.SET(a)
	} else {
		r.SET(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (r *Real) MAX(a, b *Real) Scalar {
	if a.GetValue() > b.GetValue() {
		r.SET(a)
	} else {
		r.SET(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (c *Real) ABS(a *Real) Scalar {
	if c.Sign() == -1 {
		c.NEG(a)
	} else {
		c.SET(a)
	}
	return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) NEG(a *Real) *Real {
	x := a.GetValue()
	return c.realMonadic(a, -x, -1, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) ADD(a, b *Real) *Real {
	x := a.GetValue()
	y := b.GetValue()
	return c.realDyadic(a, b, x+y, 1, 1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) SUB(a, b *Real) *Real {
	x := a.GetValue()
	y := b.GetValue()
	return c.realDyadic(a, b, x-y, 1, -1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) MUL(a, b *Real) *Real {
	x := a.GetValue()
	y := b.GetValue()
	return c.realDyadic(a, b, x*y, y, x, 1, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) DIV(a, b *Real) *Real {
	x := a.GetValue()
	y := b.GetValue()
	return c.realDyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}

/* -------------------------------------------------------------------------- */

func (c *Real) LOGADD(a, b, t *Real) *Real {
	if a.GREATER(b) {
		// swap
		a, b = b, a
	}
	if math.IsInf(a.GetValue(), 0) {
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

func (c *Real) LOGSUB(a, b, t *Real) *Real {
	if math.IsInf(b.GetValue(), -1) {
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

func (c *Real) POW(a, k *Real) *Real {
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

func (c *Real) SQRT(a *Real) *Real {
	return c.POW(a, NewReal(0.5))
}

/* -------------------------------------------------------------------------- */

func (c *Real) EXP(a *Real) *Real {
	x := a.GetValue()
	v0 := math.Exp(x)
	f1 := func() float64 { return v0 }
	f2 := func() float64 { return v0 }
	return c.realMonadicLazy(a, v0, f1, f2)
}

func (c *Real) LOG(a *Real) *Real {
	x := a.GetValue()
	v0 := math.Log(x)
	f1 := func() float64 { return 1 / x }
	f2 := func() float64 { return -1 / (x * x) }
	return c.realMonadicLazy(a, v0, f1, f2)
}

func (c *Real) LOG1P(a *Real) *Real {
	x := a.GetValue()
	v0 := math.Log1p(x)
	f1 := func() float64 { return 1 / (1 + x) }
	f2 := func() float64 { return -1 / ((1 + x) * (1 + x)) }
	return c.realMonadicLazy(a, v0, f1, f2)
}
