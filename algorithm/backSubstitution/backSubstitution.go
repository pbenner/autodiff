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

package backSubstitution

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
	A Matrix
	X Vector
	T Scalar
}

/* -------------------------------------------------------------------------- */

func backSubstitution(inSitu *InSitu, b Vector) (Vector, error) {

	A := inSitu.A
	x := inSitu.X
	t := inSitu.T

	_, n := A.Dims()

	for i := n - 1; i >= 0; i-- {
		if b == nil {
			x.At(i).SetValue(0.0)
		} else {
			x.At(i).Set(b.At(i))
		}
		for j := i + 1; j < n; j++ {
			t.Mul(A.At(i, j), x.At(j))
			x.At(i).Sub(x.At(i), t)
		}
		x.At(i).Div(x.At(i), A.At(i, i))
	}
	return x, nil
}

/* -------------------------------------------------------------------------- */

func Run(A Matrix, b Vector, args ...interface{}) (Vector, error) {
	m, n := A.Dims()
	t := A.ElementType()

	if m != n {
		return nil, fmt.Errorf("matrix must be square")
	}
	if b != nil && m != b.Dim() {
		return nil, fmt.Errorf("matrix vector dimensions do not match")
	}
	inSitu := &InSitu{}

	// loop over optional arguments
	for _, arg := range args {
		switch tmp := arg.(type) {
		case *InSitu:
			inSitu = tmp
		case InSitu:
			panic("InSitu must be passed by reference")
		}
	}
	if inSitu.A == nil {
		inSitu.A = A.CloneMatrix()
	} else {
		if n1, m1 := inSitu.A.Dims(); n1 != n || m1 != m {
			return nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
		}
	}
	if inSitu.X == nil {
		inSitu.X = NullVector(t, n)
	} else {
		if inSitu.X.Dim() != n {
			return nil, fmt.Errorf("x has invalid dimension")
		}
	}
	if inSitu.T == nil {
		inSitu.T = NullScalar(t)
	}
	return backSubstitution(inSitu, b)
}
