/* Copyright (C) 2015-2017 Philipp Benner
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

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
	L Matrix
	D Matrix
	S Scalar
	T Scalar
}

type LDL struct {
	Value bool
}

type ForcePD struct {
	Value bool
}

/* -------------------------------------------------------------------------- */

func Run(a ConstMatrix, args ...interface{}) (Matrix, Matrix, error) {
	n, m := a.Dims()
	if n != m {
		panic("Cholesky(): Not a square matrix!")
	}
	if n == 0 {
		panic("Cholesky(): Empty matrix!")
	}
	t := a.ElementType()
	inSitu := &InSitu{}
	ldl := false
	forcePD := false

	for _, arg := range args {
		switch a := arg.(type) {
		case LDL:
			ldl = a.Value
		case ForcePD:
			forcePD = a.Value
		case *InSitu:
			inSitu = a
		case InSitu:
			panic("InSitu must be passed by reference")
		default:
			panic("Cholesky(): Invalid optional argument!")
		}
	}
	// allocate memory
	if inSitu.L == nil {
		inSitu.L = NullMatrix(t, n, n)
	}
	if ldl {
		if inSitu.D == nil {
			inSitu.D = NullMatrix(t, n, n)
		} else {
			inSitu.D.Map(func(x Scalar) { x.SetValue(0.0) })
		}
	}
	if inSitu.S == nil {
		inSitu.S = NewScalar(t, 0.0)
	}
	if inSitu.T == nil {
		inSitu.T = NewScalar(t, 0.0)
	}
	if ldl {
		{ // Real
			A, ok1 := a.(*DenseRealMatrix)
			L, ok2 := inSitu.L.(*DenseRealMatrix)
			D, ok3 := inSitu.D.(*DenseRealMatrix)
			s, ok4 := inSitu.S.(*Real)
			t, ok5 := inSitu.T.(*Real)
			if ok1 && ok2 && ok3 && ok4 && ok5 {
				if forcePD {
					return cholesky_ldl_forcepd_real(A, L, D, s, t)
				} else {
					return cholesky_ldl_real(A, L, D, s, t)
				}
			}
		}
		{ // BareReal
			A, ok1 := a.(*DenseBareRealMatrix)
			L, ok2 := inSitu.L.(*DenseBareRealMatrix)
			D, ok3 := inSitu.D.(*DenseBareRealMatrix)
			s, ok4 := inSitu.S.(*BareReal)
			t, ok5 := inSitu.T.(*BareReal)
			if ok1 && ok2 && ok3 && ok4 && ok5 {
				if forcePD {
					return cholesky_ldl_forcepd_barereal(A, L, D, s, t)
				} else {
					return cholesky_ldl_barereal(A, L, D, s, t)
				}
			}
		}
		// generic
		if forcePD {
			return cholesky_ldl_forcepd(a, inSitu.L, inSitu.D, inSitu.S, inSitu.T)
		} else {
			return cholesky_ldl(a, inSitu.L, inSitu.D, inSitu.S, inSitu.T)
		}
	} else {
		{
			A, ok1 := a.(*DenseRealMatrix)
			L, ok2 := inSitu.L.(*DenseRealMatrix)
			s, ok3 := inSitu.S.(*Real)
			t, ok4 := inSitu.T.(*Real)
			if ok1 && ok2 && ok3 && ok4 {
				return cholesky_real(A, L, s, t)
			}
		}
		{
			A, ok1 := a.(*DenseBareRealMatrix)
			L, ok2 := inSitu.L.(*DenseBareRealMatrix)
			s, ok3 := inSitu.S.(*BareReal)
			t, ok4 := inSitu.T.(*BareReal)
			if ok1 && ok2 && ok3 && ok4 {
				return cholesky_barereal(A, L, s, t)
			}
		}
		return cholesky(a, inSitu.L, inSitu.S, inSitu.T)
	}
}
