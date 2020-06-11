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

type Submatrix struct {
	Value []bool
}

type UpperTriangular struct {
	Value bool
}

/* -------------------------------------------------------------------------- */

func gaussJordan(a, x Matrix, b Vector, submatrix []bool) error {
	t := NewScalar(a.ElementType(), 0.0)
	c := NewScalar(a.ElementType(), 0.0)
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
	if b.Dim() != n {
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
			if math.Abs(a.At(p[j], i).GetValue()) > math.Abs(a.At(p[maxrow], i).GetValue()) {
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
			c.Div(a.At(p[j], i), a.At(p[i], i))
			// loop over columns in a
			for k := i; k < n; k++ {
				if !submatrix[k] {
					continue
				}
				// a[j, k] -= a[i, k]*c
				t.Mul(a.At(p[i], k), c)
				a.At(p[j], k).Sub(a.At(p[j], k), t)
			}
			// loop over columns in x
			for k := 0; k < n; k++ {
				if !submatrix[k] {
					continue
				}
				// x[j, k] -= x[i, k]*c
				t.Mul(x.At(p[i], k), c)
				x.At(p[j], k).Sub(x.At(p[j], k), t)
			}
			// same for b: b[j] -= b[j]*c
			t.Mul(b.At(p[i]), c)
			b.At(p[j]).Sub(b.At(p[j]), t)
		}
	}
	// backsubstitute
	for i := n - 1; i >= 0; i-- {
		if !submatrix[i] {
			continue
		}
		c.Set(a.At(p[i], i))
		for j := 0; j < i; j++ {
			if !submatrix[j] {
				continue
			}
			// b[j] -= a[j,i]*b[i]/c
			t.Mul(a.At(p[j], i), b.At(p[i]))
			t.Div(t, c)
			b.At(p[j]).Sub(b.At(p[j]), t)
			if math.IsNaN(b.At(p[j]).GetValue()) {
				goto singular
			}
			// loop over colums in x
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// x[j,k] -= a[j,i]*x[i,k]/c
				t.Mul(a.At(p[j], i), x.At(p[i], k))
				t.Div(t, c)
				x.At(p[j], k).Sub(x.At(p[j], k), t)
				if math.IsNaN(x.At(p[j], k).GetValue()) {
					goto singular
				}
			}
			// loop over colums in a
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// a[j,k] -= a[j,i]*a[i,k]/c
				t.Mul(a.At(p[j], i), a.At(p[i], k))
				t.Div(t, c)
				a.At(p[j], k).Sub(a.At(p[j], k), t)
				if math.IsNaN(a.At(p[j], k).GetValue()) {
					goto singular
				}
			}
		}
		a.At(p[i], i).Div(a.At(p[i], i), c)
		if math.IsNaN(a.At(p[i], i).GetValue()) {
			goto singular
		}
		// normalize ith row in x
		for k := 0; k < n; k++ {
			if !submatrix[k] {
				continue
			}
			x.At(p[i], k).Div(x.At(p[i], k), c)
		}
		// normalize ith element in b
		b.At(p[i]).Div(b.At(p[i]), c)
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
	panic("system is computationally singular")
}

func gaussJordanUpperTriangular(a, x Matrix, b Vector, submatrix []bool) error {
	t := NewScalar(a.ElementType(), 0.0)
	c := NewScalar(a.ElementType(), 0.0)
	// number of rows
	n, _ := a.Dims()
	// x and b should have the same number of rows
	if m, _ := x.Dims(); m != n {
		panic("GaussJordan(): x has invalid dimension!")
	}
	if b.Dim() != n {
		panic("GaussJordan(): b has invalid dimension!")
	}
	// backsubstitute
	for i := n - 1; i >= 0; i-- {
		if !submatrix[i] {
			continue
		}
		c.Set(a.At(i, i))
		for j := 0; j < i; j++ {
			if !submatrix[j] {
				continue
			}
			// b[j] -= a[j,i]*b[i]/c
			t.Mul(a.At(j, i), b.At(i))
			t.Div(t, c)
			b.At(j).Sub(b.At(j), t)
			if math.IsNaN(b.At(j).GetValue()) {
				goto singular
			}
			// loop over colums in x
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// x[j,k] -= a[j,i]*x[i,k]/c
				t.Mul(a.At(j, i), x.At(i, k))
				t.Div(t, c)
				x.At(j, k).Sub(x.At(j, k), t)
				if math.IsNaN(x.At(j, k).GetValue()) {
					goto singular
				}
			}
			// loop over colums in a
			for k := n - 1; k >= 0; k-- {
				if !submatrix[k] {
					continue
				}
				// a[j,k] -= a[j,i]*a[i,k]/c
				t.Mul(a.At(j, i), a.At(i, k))
				t.Div(t, c)
				a.At(j, k).Sub(a.At(j, k), t)
				if math.IsNaN(a.At(j, k).GetValue()) {
					goto singular
				}
			}
		}
		a.At(i, i).Div(a.At(i, i), c)
		if math.IsNaN(a.At(i, i).GetValue()) {
			goto singular
		}
		// normalize ith row in x
		for k := i; k < n; k++ {
			if !submatrix[k] {
				continue
			}
			x.At(i, k).Div(x.At(i, k), c)
		}
		// normalize ith element in b
		b.At(i).Div(b.At(i), c)
	}
	return nil
singular:
	panic("system is computationally singular")
}

/* -------------------------------------------------------------------------- */

func Run(a, x Matrix, b Vector, args ...interface{}) error {

	submatrix := Submatrix{nil}.Value
	triangular := false

	// loop over optional arguments
	for _, arg := range args {
		switch a := arg.(type) {
		case Submatrix:
			submatrix = a.Value
		case UpperTriangular:
			triangular = a.Value
		default:
			panic("GaussJordan(): Invalid optional argument!")
		}
	}
	// initialize with default values
	if submatrix == nil {
		n, _ := a.Dims()
		submatrix = make([]bool, n)
		for i, _ := range submatrix {
			submatrix[i] = true
		}
	}
	ad, ok1 := a.(*DenseBareRealMatrix)
	bd, ok2 := b.(DenseBareRealVector)
	xd, ok3 := x.(*DenseBareRealMatrix)
	if ok1 && ok2 && ok3 {
		if triangular == true {
			return gaussJordanUpperTriangular_DenseBareReal(ad, xd, bd, submatrix)
		} else {
			return gaussJordan_DenseBareReal(ad, xd, bd, submatrix)
		}
	}
	// call generic gaussJordan
	if triangular {
		return gaussJordanUpperTriangular(a, x, b, submatrix)
	} else {
		return gaussJordan(a, x, b, submatrix)
	}
}
