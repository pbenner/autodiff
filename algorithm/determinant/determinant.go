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

package determinant

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/cholesky"

/* -------------------------------------------------------------------------- */

type PositiveDefinite struct {
	Value bool
}

type LogScale struct {
	Value bool
}

type InSitu struct {
	Cholesky cholesky.InSitu
}

/* -------------------------------------------------------------------------- */

func determinantNaive(a ConstMatrix) Scalar {
	n, _ := a.Dims()
	t1 := NullScalar(a.ElementType())
	t2 := NullScalar(a.ElementType())
	det := NullScalar(a.ElementType())

	if n < 1 {
		/* nothing to do */
	} else if n == 1 {
		det.Set(a.ConstAt(0, 0))
	} else if n == 2 {
		t1.Mul(a.ConstAt(0, 0), a.ConstAt(1, 1))
		t2.Mul(a.ConstAt(1, 0), a.ConstAt(0, 1))
		det.Sub(t1, t2)
	} else {
		m := NullMatrix(a.ElementType(), n-1, n-1)
		for j1 := 0; j1 < n; j1++ {
			for i := 1; i < n; i++ {
				j2 := 0
				for j := 0; j < n; j++ {
					if j == j1 {
						continue
					}
					m.At(i-1, j2).Set(a.ConstAt(i, j))
					j2++
				}
			}
			if j1%2 == 0 {
				t1.Mul(a.ConstAt(0, j1), determinantNaive(m))
				det.Add(det, t1)
			} else {
				t1.Mul(a.ConstAt(0, j1), determinantNaive(m))
				det.Sub(det, t1)
			}
		}
	}
	return det
}

func determinantPD(a ConstMatrix, logScale bool, inSitu *InSitu) (Scalar, error) {
	n, m := a.Dims()
	if n != m {
		panic("Matrix is not a square matrix!")
	}
	L, _, err := cholesky.Run(a, &inSitu.Cholesky)
	if err != nil {
		return nil, err
	}
	r := NullScalar(L.ElementType())
	t := NullScalar(L.ElementType())
	if logScale {
		r.SetValue(0.0)
		for i := 0; i < n; i++ {
			t.Log(L.At(i, i))
			r.Add(r, t)
		}
		r.Add(r, r)
	} else {
		r.SetValue(1.0)
		for i := 0; i < n; i++ {
			r.Mul(r, L.At(i, i))
		}
		r.Mul(r, r)
	}
	return r, nil
}

func determinant(a ConstMatrix, positiveDefinite, logScale bool, inSitu *InSitu) (Scalar, error) {
	if positiveDefinite {
		return determinantPD(a, logScale, inSitu)
	} else {
		return determinantNaive(a), nil
	}
}

/* -------------------------------------------------------------------------- */

func Run(a ConstMatrix, args ...interface{}) (Scalar, error) {
	positiveDefinite := false
	logScale := false
	inSitu := &InSitu{}

	// loop over optional arguments
	for _, arg := range args {
		switch a := arg.(type) {
		case PositiveDefinite:
			positiveDefinite = a.Value
		case LogScale:
			logScale = a.Value
		case *InSitu:
			inSitu = a
		case InSitu:
			panic("InSitu must be passed by reference")
		default:
			panic("Determinant(): Invalid optional argument!")
		}
	}
	if logScale && !positiveDefinite {
		panic("Parameter LogScale is valid only for positive definite matrices!")
	}
	return determinant(a, positiveDefinite, logScale, inSitu)
}
