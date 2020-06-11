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

package gaussJordan

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "errors"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func gaussJordan_DenseBareReal(a, x *DenseBareRealMatrix, b DenseBareRealVector, submatrix []bool) error {
	t := NewBareReal(0.0)
	c := NewBareReal(0.0)
	// number of rows
	n, _ := a.Dims()
	// permutation of the rows
	p := make([]int, n)
	for i := 0; i < n; i++ {
		p[i] = i
	}
	// x and b should have the same number of rows
	if m, _ := x.Dims(); m != n {
		return errors.New("GaussJordan(): x has invalid dimension!")
	}
	if len(b) != n {
		return errors.New("GaussJordan(): b has invalid dimension!")
	}
	// loop over columns
	for i := 0; i < n; i++ {
		if !submatrix[i] {
			continue
		}
		// find row with maximum value at column i
		maxrow := i
		for j := i + 1; j < n; j++ {
			if !submatrix[j] {
				continue
			}
			if math.Abs(a.AT(p[j], i).GetValue()) > math.Abs(a.AT(p[maxrow], i).GetValue()) {
				maxrow = j
			}
		}
		// swap rows
		p[i], p[maxrow] = p[maxrow], p[i]
		// eliminate column i
		for j := i + 1; j < n; j++ {
			if !submatrix[j] {
				continue
			}
			// c = a[j, i] / a[i, i]
			c.DIV(a.AT(p[j], i), a.AT(p[i], i))
			// loop over columns in a
			for k := i; k < n; k++ {
				if !submatrix[k] {
					continue
				}
				// a[j, k] -= a[i, k]*c
				t.MUL(a.AT(p[i], k), c)
				a.AT(p[j], k).SUB(a.AT(p[j], k), t)
			}
			// loop over columns in x
			for k := 0; k < n; k++ {
				if !submatrix[k] {
					continue
				}
				// x[j, k] -= x[i, k]*c
				t.MUL(x.AT(p[i], k), c)
				x.AT(p[j], k).SUB(x.AT(p[j], k), t)
			}
			// same for b: b[j] -= b[j]*c
			t.MUL(b.AT(p[i]), c)
			b.AT(p[j]).SUB(b.AT(p[j]), t)
		}
	}
	// backsubstitute
	for i := n - 1; i >= 0; i-- {
		if !submatrix[i] {
			continue
		}
		c.Set(a.AT(p[i], i))
		for j := 0; j < i; j++ {
			if !submatrix[j] {
				continue
			}
			// b[j] -= a[j,i]*b[i]/c
			t.MUL(a.AT(p[j], i), b.AT(p[i]))
			t.DIV(t, c)
			b.AT(p[j]).SUB(b.AT(p[j]), t)
			if math.IsNaN(b.AT(p[j]).GetValue()) {
				goto singular
			}
			// loop over colums in x
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// x[j,k] -= a[j,i]*x[i,k]/c
				t.MUL(a.AT(p[j], i), x.AT(p[i], k))
				t.DIV(t, c)
				x.AT(p[j], k).SUB(x.AT(p[j], k), t)
				if math.IsNaN(x.AT(p[j], k).GetValue()) {
					goto singular
				}
			}
			// loop over colums in a
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// a[j,k] -= a[j,i]*a[i,k]/c
				t.MUL(a.AT(p[j], i), a.AT(p[i], k))
				t.DIV(t, c)
				a.AT(p[j], k).SUB(a.AT(p[j], k), t)
				if math.IsNaN(a.AT(p[j], k).GetValue()) {
					goto singular
				}
			}
		}
		a.AT(p[i], i).DIV(a.AT(p[i], i), c)
		if math.IsNaN(a.AT(p[i], i).GetValue()) {
			goto singular
		}
		// normalize ith row in x
		for k := 0; k < n; k++ {
			if !submatrix[k] {
				continue
			}
			x.AT(p[i], k).DIV(x.AT(p[i], k), c)
		}
		// normalize ith element in b
		b.AT(p[i]).DIV(b.AT(p[i]), c)
	}
	if err := a.PermuteRows(p); err != nil {
		return err
	}
	if err := x.PermuteRows(p); err != nil {
		return err
	}
	if err := b.Permute(p); err != nil {
		return err
	}
	return nil
singular:
	return errors.New("system is computationally singular")
}

func gaussJordanUpperTriangular_DenseBareReal(a, x *DenseBareRealMatrix, b DenseBareRealVector, submatrix []bool) error {
	t := NewBareReal(0.0)
	c := NewBareReal(0.0)
	// number of rows
	n, _ := a.Dims()
	// x and b should have the same number of rows
	if m, _ := x.Dims(); m != n {
		return errors.New("GaussJordan(): x has invalid dimension!")
	}
	if len(b) != n {
		return errors.New("GaussJordan(): b has invalid dimension!")
	}
	// backsubstitute
	for i := n - 1; i >= 0; i-- {
		if !submatrix[i] {
			continue
		}
		c.Set(a.AT(i, i))
		for j := 0; j < i; j++ {
			if !submatrix[j] {
				continue
			}
			// b[j] -= a[j,i]*b[i]/c
			t.MUL(a.AT(j, i), b.AT(i))
			t.DIV(t, c)
			b.AT(j).SUB(b.AT(j), t)
			if math.IsNaN(b.AT(j).GetValue()) {
				goto singular
			}
			// loop over colums in x
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// x[j,k] -= a[j,i]*x[i,k]/c
				t.MUL(a.AT(j, i), x.AT(i, k))
				t.DIV(t, c)
				x.AT(j, k).SUB(x.AT(j, k), t)
				if math.IsNaN(x.AT(j, k).GetValue()) {
					goto singular
				}
			}
			// loop over colums in a
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// a[j,k] -= a[j,i]*a[i,k]/c
				t.MUL(a.AT(j, i), a.AT(i, k))
				t.DIV(t, c)
				a.AT(j, k).SUB(a.AT(j, k), t)
				if math.IsNaN(a.AT(j, k).GetValue()) {
					goto singular
				}
			}
		}
		a.AT(i, i).DIV(a.AT(i, i), c)
		if math.IsNaN(a.AT(i, i).GetValue()) {
			goto singular
		}
		// normalize ith row in x
		for k := i; k < n; k++ {
			if !submatrix[k] {
				continue
			}
			x.AT(i, k).DIV(x.AT(i, k), c)
		}
		// normalize ith element in b
		b.AT(i).DIV(b.AT(i), c)
	}
	return nil
singular:
	return errors.New("system is computationally singular")
}
