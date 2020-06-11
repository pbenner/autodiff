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

package hessenbergReduction

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/householder"

/* -------------------------------------------------------------------------- */

type SetZero struct {
	Value bool
}

type ComputeU struct {
	Value bool
}

type InSitu struct {
	H    Matrix
	U    Matrix
	X    Vector
	Beta Scalar
	Nu   Vector
	T1   Scalar
	T2   Scalar
	T3   Scalar
	T4   Vector
}

/* -------------------------------------------------------------------------- */

func hessenbergReduction(inSitu *InSitu, setZero bool) (Matrix, Matrix, error) {
	H := inSitu.H
	U := inSitu.U
	x := inSitu.X
	beta := inSitu.Beta
	nu := inSitu.Nu
	t1 := inSitu.T1
	t2 := inSitu.T2
	t3 := inSitu.T3
	t4 := inSitu.T4

	n, _ := H.Dims()

	for k := 0; k < n-2; k++ {
		// copy column below main diagonal from H to x,
		// x = (H[k+1,k], H[k+2,k], ..., H[n-1,k])
		for i := k + 1; i < n; i++ {
			x.At(i).Set(H.At(i, k))
		}
		householder.Run(x.Slice(k+1, n), beta, nu.Slice(k+1, n), t1, t2, t3)
		{
			a := H.Slice(k+1, n, k, n)
			householder.ApplyLeft(a, beta, nu.Slice(k+1, n), t4.Slice(k, n), t1)
		}
		{
			a := H.Slice(0, n, k+1, n)
			householder.ApplyRight(a, beta, nu.Slice(k+1, n), t4.Slice(0, n), t1)
		}
		if setZero {
			for i := k + 2; i < n; i++ {
				H.At(i, k).SetValue(0.0)
			}
		}
		if U != nil {
			nu.At(k).SetValue(0.0)
			householder.ApplyRight(U, beta, nu.Slice(0, n), t4.Slice(0, n), t1)
		}
	}
	return H, U, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

	n, m := a.Dims()
	t := a.ElementType()

	if n != m {
		return nil, nil, fmt.Errorf("`a' must be a square matrix")
	}
	inSitu := &InSitu{}
	computeU := false
	setZero := true

	// loop over optional arguments
	for _, arg := range args {
		switch tmp := arg.(type) {
		case ComputeU:
			computeU = tmp.Value
		case SetZero:
			setZero = tmp.Value
		case *InSitu:
			inSitu = tmp
		case InSitu:
			panic("InSitu must be passed by reference")
		}
	}
	if inSitu.H == nil {
		inSitu.H = a.CloneMatrix()
	} else {
		if inSitu.H != a {
			inSitu.H.Set(a)
		}
	}
	if inSitu.Beta == nil {
		inSitu.Beta = NullScalar(t)
	}
	if inSitu.Nu == nil {
		inSitu.Nu = NullVector(t, n)
	}
	if inSitu.X == nil {
		inSitu.X = NullVector(t, n)
	}
	if computeU {
		if inSitu.U == nil {
			inSitu.U = NullMatrix(t, n, n)
		}
		inSitu.U.SetIdentity()
	} else {
		inSitu.U = nil
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
		inSitu.T4 = NullVector(t, n)
	}
	return hessenbergReduction(inSitu, setZero)
}
