/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2017 Philipp Benner
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
package cholesky

/* -------------------------------------------------------------------------- */
import "fmt"
import "math"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */
func cholesky(A ConstMatrix, L Matrix, s, t Scalar) (Matrix, Matrix, error) {
	n, _ := A.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < (i + 1); j++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.Mul(L.At(i, k), L.At(j, k))
				s.Add(s, t)
			}
			t.Sub(A.ConstAt(i, j), s)
			if i == j {
				if t.GetValue() < 0.0 {
					return nil, nil, fmt.Errorf("matrix is not positive definite")
				}
				L.At(i, j).Sqrt(t)
			} else {
				L.At(i, j).Div(t, L.At(j, j))
			}
		}
	}
	return L, nil, nil
}
func cholesky_ldl(A ConstMatrix, L, D Matrix, s, t Scalar) (Matrix, Matrix, error) {
	n, _ := A.Dims()
	c := t
	for j := 0; j < n; j++ {
		// compute diagonal entries
		s.Reset()
		for k := 0; k < j; k++ {
			t.Mul(L.At(j, k), L.At(j, k))
			t.Mul(D.At(k, k), t)
			s.Add(s, t)
		}
		c.Sub(A.ConstAt(j, j), s)
		D.At(j, j).Set(c)
		if D.At(j, j).GetValue() <= 0.0 {
			return nil, nil, fmt.Errorf("matrix is not positive definite")
		}
		L.At(j, j).SetValue(1.0)
		// compute remaining entries
		for i := j + 1; i < n; i++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.Mul(L.At(i, k), L.At(j, k))
				t.Mul(D.At(k, k), t)
				s.Add(s, t)
			}
			c.Sub(A.ConstAt(i, j), s)
			L.At(i, j).Div(c, D.At(j, j))
		}
	}
	return L, D, nil
}
func cholesky_ldl_forcepd(A ConstMatrix, L, D Matrix, s, t Scalar) (Matrix, Matrix, error) {
	n, _ := A.Dims()
	// compute beta and gamma
	beta := 0.0
	gamma := math.Inf(-1)
	nu := math.Max(1, math.Sqrt(float64(n*n-1)))
	theta := math.Inf(-1)
	xi := math.Inf(-1)
	delta := 1e-20
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				if r := math.Abs(A.ConstAt(i, i).GetValue()); r > gamma {
					gamma = r
				}
			} else {
				if r := math.Abs(A.ConstAt(i, j).GetValue()); r > xi {
					xi = r
				}
			}
		}
	}
	beta = math.Max(gamma, xi/nu)
	beta = math.Max(beta, 1e-20)
	beta = math.Sqrt(beta)
	// loop over columns
	for j := 0; j < n; j++ {
		L.At(j, j).SetValue(1.0)
		// compute c_jj (stored temporarily in d_j)
		s.Reset()
		for k := 0; k < j; k++ {
			t.Mul(L.At(j, k), L.At(j, k))
			t.Mul(D.At(k, k), t)
			s.Add(s, t)
		}
		c_jj := D.At(j, j)
		c_jj.Sub(A.ConstAt(j, j), s)
		// reset theta_j
		theta = math.Inf(-1)
		// compute c_ij and theta_j
		for i := j + 1; i < n; i++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.Mul(L.At(i, k), L.At(j, k))
				t.Mul(D.At(k, k), t)
				s.Add(s, t)
			}
			// result: L(i,j) <- c_ij
			L.At(i, j).Sub(A.ConstAt(i, j), s)
			// update theta_j
			if r := math.Abs(L.At(i, j).GetValue()); r > theta {
				theta = r
			}
		}
		// compute d_j = max(|c_jj|, (theta_j/beta)^2, delta)
		if j != n-1 {
			D.At(j, j).SetValue(
				math.Max(math.Max(math.Abs(c_jj.GetValue()), math.Pow((theta/beta), 2.0)), delta))
		} else {
			D.At(j, j).SetValue(
				math.Max(math.Abs(c_jj.GetValue()), delta))
		}
		// compute l_ij = c_ij/d_j
		for i := j + 1; i < n; i++ {
			L.At(i, j).Div(L.At(i, j), D.At(j, j))
		}
	}
	return L, D, nil
}
