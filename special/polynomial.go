/* Copyright (C) 2016 Philipp Benner
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

package special

/* -------------------------------------------------------------------------- */

type Polynomial struct {
  coefficients []float64
}

type EvenPolynomial struct {
  Polynomial
}

/* -------------------------------------------------------------------------- */

func NewPolynomial(coefficients []float64) Polynomial {
  return Polynomial{coefficients}
}

func NewEvenPolynomial(coefficients []float64) EvenPolynomial {
  return EvenPolynomial{NewPolynomial(coefficients)}
}

/* -------------------------------------------------------------------------- */

func (p Polynomial) Eval(z float64) float64 {
  count := len(p.coefficients)
  sum := p.coefficients[count - 1];
  for i := count - 2; i >= 0; i-- {
    sum *= z;
    sum += p.coefficients[i];
  }
  return sum;
}

func (p EvenPolynomial) Eval(z float64) float64 {
  return p.Polynomial.Eval(z*z)
}
