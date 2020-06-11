/* Copyright (C) 2017 Philipp Benner
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

package householder

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func ApplyLeft(A Matrix, beta Scalar, nu Vector, t1 Vector, t2 Scalar) {
	m, n := A.Dims()

	t1.VdotM(nu, A)
	t1.VmulS(t1, beta)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			t2.Mul(nu.At(i), t1.At(j))

			a := A.At(i, j)
			a.Sub(a, t2)
		}
	}
}

func ApplyRight(A Matrix, beta Scalar, nu Vector, t1 Vector, t2 Scalar) {
	m, n := A.Dims()

	t1.MdotV(A, nu)
	t1.VmulS(t1, beta)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			t2.Mul(t1.At(i), nu.At(j))

			a := A.At(i, j)
			a.Sub(a, t2)
		}
	}
}

/* -------------------------------------------------------------------------- */

func Run(x Vector, beta Scalar, nu Vector, t1, t2, t3 Scalar) {
	sigma := t1
	mu := t2
	t := t3
	sigma.SetValue(0.0)
	nu.At(0).SetValue(1.0)
	for i := 1; i < x.Dim(); i++ {
		nu.At(i).Set(x.At(i))
		t.Mul(x.At(i), x.At(i))
		sigma.Add(sigma, t)
	}
	if sigma.GetValue() == 0.0 {
		beta.SetValue(0.0)
	} else {
		nu0 := nu.At(0)
		mu.Mul(x.At(0), x.At(0))
		mu.Add(mu, sigma)
		mu.Sqrt(mu)
		if x.At(0).GetValue() <= 0.0 {
			nu0.Sub(x.At(0), mu)
		} else {
			nu0.Add(x.At(0), mu)
			nu0.Div(sigma, nu0)
			nu0.Neg(nu0)
		}
		beta.Mul(nu0, nu0)
		beta.Add(beta, sigma)
		beta.Div(nu0, beta)
		beta.Mul(beta, nu0)
		beta.Add(beta, beta)
		t.Set(nu0)
		nu.VdivS(nu, t)
	}
}
