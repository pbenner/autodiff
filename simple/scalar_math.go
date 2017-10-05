/* Copyright (C) 2015, 2016, 2017 Philipp Benner
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

package simple

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Equal(a, b Scalar, epsilon float64) bool {
  return a.Equals(b, epsilon)
}

func Greater(a, b Scalar) bool {
  return a.Greater(b)
}

func Smaller(a, b Scalar) bool {
  return a.Smaller(b)
}

func Neg(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Neg(a)
}

func Abs(a Scalar) Scalar {
  c := a.CloneScalar()
  if c.Sign() == -1 {
    c.Neg(c)
  }
  return c
}

func Add(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Add(a, b)
}

func Sub(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Sub(a, b)
}

func Mul(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Mul(a, b)
}

func Div(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Div(a, b)
}

func Pow(a Scalar, k Scalar) Scalar {
  c := a.CloneScalar()
  return c.Pow(a, k)
}

func Sqrt(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Sqrt(a)
}

func Sin(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Sin(a)
}

func Sinh(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Sinh(a)
}

func Cos(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Cos(a)
}

func Cosh(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Cosh(a)
}

func Tan(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Tan(a)
}

func Tanh(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Tanh(a)
}

func Exp(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Exp(a)
}

func Log(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Log(a)
}

func Erf(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Erf(a)
}

func Erfc(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Erfc(a)
}

func LogErfc(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.LogErfc(a)
}

func Gamma(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Gamma(a)
}

func Lgamma(a Scalar) Scalar {
  c := a.CloneScalar()
  return c.Lgamma(a)
}

func Mlgamma(a Scalar, k int) Scalar {
  c := a.CloneScalar()
  return c.Mlgamma(a, k)
}

func GammaP(a float64, x Scalar) Scalar {
  c := x.CloneScalar()
  return c.GammaP(a, x)
}

func Min(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Min(a, b)
}

func Max(a, b Scalar) Scalar {
  c := a.CloneScalar()
  return c.Max(a, b)
}

func Vnorm(a Vector) Scalar {
  r := NullScalar(a.ElementType())
  r.Vnorm(a)
  return r
}
