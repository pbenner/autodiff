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

package householderTridiagonalization

/* -------------------------------------------------------------------------- */

import "fmt"

//import   "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/householder"

/* -------------------------------------------------------------------------- */

type ComputeU struct {
	Value bool
}

/* -------------------------------------------------------------------------- */

type Epsilon struct {
	Value float64
}

type InSitu struct {
	A    Matrix
	U    Matrix
	V    Matrix
	X    Vector
	Beta Scalar
	Nu   Vector
	C1   Scalar
	T1   Scalar
	T2   Scalar
	T3   Scalar
	T4   Vector
}

/* -------------------------------------------------------------------------- */

func houseCol(k int, inSitu *InSitu) (Vector, Scalar) {
	A := inSitu.A
	x := inSitu.X
	beta := inSitu.Beta
	nu := inSitu.Nu
	t1 := inSitu.T1
	t2 := inSitu.T2
	t3 := inSitu.T3

	n, _ := A.Dims()
	for j := k + 1; j < n; j++ {
		x.At(j).Set(A.At(j, k))
	}
	householder.Run(x.Slice(k+1, n), beta, nu.Slice(k+1, n), t1, t2, t3)
	return nu.Slice(k+1, n), beta
}

/* -------------------------------------------------------------------------- */

func householderTridiagonalization(inSitu *InSitu, epsilon float64) (Matrix, Matrix, error) {

	A := inSitu.A
	U := inSitu.U
	t := inSitu.T1
	s := inSitu.T2
	p := inSitu.X
	w := inSitu.T4
	c := BareReal(2.0)

	_, n := A.Dims()

	for k := 0; k < n-2; k++ {

		nu, beta := houseCol(k, inSitu)

		a := A.Slice(k+1, n, k+1, n)
		p := p.Slice(k+1, n)
		w := w.Slice(k+1, n)

		p.MdotV(a, nu)
		p.VmulS(p, beta)

		t.VdotV(p, nu)
		t.Mul(t, beta)
		t.Div(t, &c)
		w.VmulS(nu, t)
		w.VsubV(p, w)

		// compute ||A(k+1:n,k)||_2
		s.SetValue(0.0)
		for j := k + 1; j < n; j++ {
			t.Mul(A.At(j, k), A.At(j, k))
			s.Add(s, t)
		}
		s.Sqrt(s)

		A.At(k+1, k+0).Set(s)
		A.At(k+0, k+1).Set(s)

		for j := k + 1; j < n; j++ {
			for i := k + 1; i < n; i++ {
				a := A.At(i, j)
				t.Mul(nu.At(i-k-1), w.At(j-k-1))
				s.Mul(nu.At(j-k-1), w.At(i-k-1))
				a.Sub(a, t)
				a.Sub(a, s)
			}
		}
		for j := k + 2; j < n; j++ {
			A.At(k, j).SetValue(0.0)
			A.At(j, k).SetValue(0.0)
		}

		if U != nil {
			u := U.Slice(0, n, k+1, n)
			householder.ApplyRight(u, beta, nu, inSitu.T4.Slice(0, n), inSitu.T1)
		}
	}
	return A, U, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

	m, n := a.Dims()
	t := a.ElementType()

	if m != n {
		return nil, nil, fmt.Errorf("`a' must be a square matrix")
	}
	inSitu := &InSitu{}
	computeU := false
	epsilon := 1.11e-16

	// loop over optional arguments
	for _, arg := range args {
		switch tmp := arg.(type) {
		case ComputeU:
			computeU = tmp.Value
		case Epsilon:
			epsilon = tmp.Value
		case *InSitu:
			inSitu = tmp
		case InSitu:
			panic("InSitu must be passed by reference")
		}
	}
	if inSitu.A == nil {
		inSitu.A = a.CloneMatrix()
	} else {
		if inSitu.A != a {
			inSitu.A.Set(a)
		}
	}
	if computeU {
		if inSitu.U == nil {
			inSitu.U = NullMatrix(t, m, m)
		}
		inSitu.U.SetIdentity()
	} else {
		inSitu.U = nil
	}
	if inSitu.X == nil {
		inSitu.X = NullVector(t, m)
	}
	if inSitu.Beta == nil {
		inSitu.Beta = NullScalar(t)
	}
	if inSitu.Nu == nil {
		inSitu.Nu = NullVector(t, m)
	}
	if inSitu.C1 == nil {
		inSitu.C1 = NewScalar(t, 1.0)
	}
	if inSitu.T1 == nil {
		inSitu.T1 = NullScalar(t)
	}
	if inSitu.T2 == nil {
		inSitu.T2 = NullScalar(t)
	}
	if inSitu.T3 == nil {
		inSitu.T3 = NullScalar(t)
	}
	if inSitu.T4 == nil {
		inSitu.T4 = NullVector(t, m)
	}
	return householderTridiagonalization(inSitu, epsilon)
}
