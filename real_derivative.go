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

/* derivatives of monadic functions
 * -------------------------------------------------------------------------- */

// Compute d/dx f(g(x)) and d^2/dx^2 f(g(x)) evaluated at x=x0, where
// - a  = g(x0)
// - v0 = f(a)
// - v1 = d/dx f(x) | x=a
// - v2 = d^2/dx^2 f(x) | x=a
func (c *Real) monadic(a ConstScalar, v0, v1, v2 float64) *Real {
	c.AllocForOne(a)
	if c.Order >= 1 {
		if c.Order >= 2 {
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetDerivative(i)*a.GetDerivative(j)*v2+
							a.GetHessian(i, j)*v1)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v1)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) monadicLazy(a ConstScalar, v0 float64, f1, f2 func() float64) *Real {
	c.AllocForOne(a)
	if c.Order >= 1 {
		v1 := f1()
		if c.Order >= 2 {
			v2 := f2()
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetDerivative(i)*a.GetDerivative(j)*v2+
							a.GetHessian(i, j)*v1)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v1)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) realMonadic(a *Real, v0, v1, v2 float64) *Real {
	c.AllocForOne(a)
	if c.Order >= 1 {
		if c.Order >= 2 {
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetDerivative(i)*a.GetDerivative(j)*v2+
							a.GetHessian(i, j)*v1)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v1)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) realMonadicLazy(a *Real, v0 float64, f1, f2 func() float64) *Real {
	c.AllocForOne(a)
	if c.Order >= 1 {
		v1 := f1()
		if c.Order >= 2 {
			v2 := f2()
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetDerivative(i)*a.GetDerivative(j)*v2+
							a.GetHessian(i, j)*v1)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v1)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

/* derivatives of dyadic functions
 * -------------------------------------------------------------------------- */

func (c *Real) dyadic(a, b ConstScalar, v0, v10, v01, v11, v20, v02 float64) *Real {
	c.AllocForTwo(a, b)
	if c.Order >= 1 {
		if c.Order >= 2 {
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetHessian(i, j)*v10+
							b.GetHessian(i, j)*v01+
							a.GetDerivative(i)*a.GetDerivative(j)*v20+
							b.GetDerivative(i)*b.GetDerivative(j)*v02+
							a.GetDerivative(i)*b.GetDerivative(j)*v11+
							b.GetDerivative(i)*a.GetDerivative(j)*v11)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v10+b.GetDerivative(i)*v01)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) dyadicLazy(a, b ConstScalar, v0 float64, f1 func() (float64, float64), f2 func() (float64, float64, float64)) *Real {
	c.AllocForTwo(a, b)
	if c.Order >= 1 {
		v10, v01 := f1()
		if c.Order >= 2 {
			v11, v20, v02 := f2()
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetHessian(i, j)*v10+
							b.GetHessian(i, j)*v01+
							a.GetDerivative(i)*a.GetDerivative(j)*v20+
							b.GetDerivative(i)*b.GetDerivative(j)*v02+
							a.GetDerivative(i)*b.GetDerivative(j)*v11+
							b.GetDerivative(i)*a.GetDerivative(j)*v11)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v10+b.GetDerivative(i)*v01)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) realDyadic(a, b *Real, v0, v10, v01, v11, v20, v02 float64) *Real {
	c.AllocForTwo(a, b)
	if c.Order >= 1 {
		if c.Order >= 2 {
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetHessian(i, j)*v10+
							b.GetHessian(i, j)*v01+
							a.GetDerivative(i)*a.GetDerivative(j)*v20+
							b.GetDerivative(i)*b.GetDerivative(j)*v02+
							a.GetDerivative(i)*b.GetDerivative(j)*v11+
							b.GetDerivative(i)*a.GetDerivative(j)*v11)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v10+b.GetDerivative(i)*v01)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}

func (c *Real) realDyadicLazy(a, b ConstScalar, v0 float64, f1 func() (float64, float64), f2 func() (float64, float64, float64)) *Real {
	c.AllocForTwo(a, b)
	if c.Order >= 1 {
		v10, v01 := f1()
		if c.Order >= 2 {
			v11, v20, v02 := f2()
			// compute hessian
			for i := 0; i < c.GetN(); i++ {
				for j := i; j < c.GetN(); j++ {
					c.SetHessian(i, j,
						a.GetHessian(i, j)*v10+
							b.GetHessian(i, j)*v01+
							a.GetDerivative(i)*a.GetDerivative(j)*v20+
							b.GetDerivative(i)*b.GetDerivative(j)*v02+
							a.GetDerivative(i)*b.GetDerivative(j)*v11+
							b.GetDerivative(i)*a.GetDerivative(j)*v11)
					c.SetHessian(j, i, c.GetHessian(i, j))
				}
			}
		}
		// compute first derivatives
		for i := 0; i < c.GetN(); i++ {
			c.SetDerivative(i, a.GetDerivative(i)*v10+b.GetDerivative(i)*v01)
		}
	}
	// compute new value
	c.setValue(v0)
	return c
}
