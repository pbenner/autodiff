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

//import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func (a *BareReal) EQUALS(b ConstScalar, epsilon float64) bool {
	return math.Abs(a.GetValue()-b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) GREATER(b *BareReal) bool {
	return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) SMALLER(b *BareReal) bool {
	return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) MIN(a, b *BareReal) Scalar {
	if a.GetValue() < b.GetValue() {
		r.SET(a)
	} else {
		r.SET(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) MAX(a, b *BareReal) Scalar {
	if a.GetValue() > b.GetValue() {
		r.SET(a)
	} else {
		r.SET(b)
	}
	return r
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) SIGN() int {
	if a.GetValue() < 0.0 {
		return -1
	}
	if a.GetValue() > 0.0 {
		return 1
	}
	return 0
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) ABS(a *BareReal) Scalar {
	if c.Sign() == -1 {
		c.NEG(a)
	} else {
		c.SET(a)
	}
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) NEG(a *BareReal) *BareReal {
	*c = BareReal(-a.GetValue())
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) ADD(a, b *BareReal) *BareReal {
	*c = *a + *b
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) SUB(a, b *BareReal) *BareReal {
	*c = *a - *b
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) MUL(a, b *BareReal) *BareReal {
	*c = *a * *b
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) DIV(a, b *BareReal) *BareReal {
	*c = *a / *b
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) LOGADD(a, b, t *BareReal) *BareReal {
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

func (c *BareReal) LOGSUB(a, b, t *BareReal) *BareReal {
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

func (c *BareReal) POW(a, k *BareReal) *BareReal {
	*c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
	return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) SQRT(a *BareReal) *BareReal {
	return c.POW(a, NewBareReal(1.0/2.0))
}

func (c *BareReal) EXP(a *BareReal) *BareReal {
	checkBare(a)
	*c = BareReal(math.Exp(a.GetValue()))
	return c
}

func (c *BareReal) LOG(a *BareReal) *BareReal {
	checkBare(a)
	*c = BareReal(math.Log(a.GetValue()))
	return c
}

func (c *BareReal) LOG1P(a *BareReal) *BareReal {
	checkBare(a)
	*c = BareReal(math.Log1p(a.GetValue()))
	return c
}
