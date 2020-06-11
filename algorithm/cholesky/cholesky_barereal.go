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
func cholesky_barereal(A *DenseBareRealMatrix, L *DenseBareRealMatrix, s, t *BareReal) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
	n, _ := A.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < (i + 1); j++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.MUL(L.AT(i, k), L.AT(j, k))
				s.ADD(s, t)
			}
			t.SUB(A.AT(i, j), s)
			if i == j {
				if t.GetValue() < 0.0 {
					return nil, nil, fmt.Errorf("matrix is not positive definite")
				}
				L.AT(i, j).SQRT(t)
			} else {
				L.AT(i, j).DIV(t, L.AT(j, j))
			}
		}
	}
	return L, nil, nil
}
func cholesky_ldl_barereal(A *DenseBareRealMatrix, L, D *DenseBareRealMatrix, s, t *BareReal) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
	n, _ := A.Dims()
	c := t
	for j := 0; j < n; j++ {
		// compute diagonal entries
		s.Reset()
		for k := 0; k < j; k++ {
			t.MUL(L.AT(j, k), L.AT(j, k))
			t.MUL(D.AT(k, k), t)
			s.ADD(s, t)
		}
		c.SUB(A.AT(j, j), s)
		D.AT(j, j).Set(c)
		if D.AT(j, j).GetValue() <= 0.0 {
			return nil, nil, fmt.Errorf("matrix is not positive definite")
		}
		L.AT(j, j).SetValue(1.0)
		// compute remaining entries
		for i := j + 1; i < n; i++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.MUL(L.AT(i, k), L.AT(j, k))
				t.MUL(D.AT(k, k), t)
				s.ADD(s, t)
			}
			c.SUB(A.AT(i, j), s)
			L.AT(i, j).DIV(c, D.AT(j, j))
		}
	}
	return L, D, nil
}
func cholesky_ldl_forcepd_barereal(A *DenseBareRealMatrix, L, D *DenseBareRealMatrix, s, t *BareReal) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
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
				if r := math.Abs(A.AT(i, i).GetValue()); r > gamma {
					gamma = r
				}
			} else {
				if r := math.Abs(A.AT(i, j).GetValue()); r > xi {
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
		L.AT(j, j).SetValue(1.0)
		// compute c_jj (stored temporarily in d_j)
		s.Reset()
		for k := 0; k < j; k++ {
			t.MUL(L.AT(j, k), L.AT(j, k))
			t.MUL(D.AT(k, k), t)
			s.ADD(s, t)
		}
		c_jj := D.AT(j, j)
		c_jj.SUB(A.AT(j, j), s)
		// reset theta_j
		theta = math.Inf(-1)
		// compute c_ij and theta_j
		for i := j + 1; i < n; i++ {
			s.Reset()
			for k := 0; k < j; k++ {
				t.MUL(L.AT(i, k), L.AT(j, k))
				t.MUL(D.AT(k, k), t)
				s.ADD(s, t)
			}
			// result: L(i,j) <- c_ij
			L.AT(i, j).SUB(A.AT(i, j), s)
			// update theta_j
			if r := math.Abs(L.AT(i, j).GetValue()); r > theta {
				theta = r
			}
		}
		// compute d_j = max(|c_jj|, (theta_j/beta)^2, delta)
		if j != n-1 {
			D.AT(j, j).SetValue(
				math.Max(math.Max(math.Abs(c_jj.GetValue()), math.Pow((theta/beta), 2.0)), delta))
		} else {
			D.AT(j, j).SetValue(
				math.Max(math.Abs(c_jj.GetValue()), delta))
		}
		// compute l_ij = c_ij/d_j
		for i := j + 1; i < n; i++ {
			L.AT(i, j).DIV(L.AT(i, j), D.AT(j, j))
		}
	}
	return L, D, nil
}
