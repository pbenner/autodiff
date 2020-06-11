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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/givensRotation"
import "github.com/pbenner/autodiff/algorithm/hessenbergReduction"
import "github.com/pbenner/autodiff/algorithm/householder"
import "github.com/pbenner/autodiff/algorithm/householderTridiagonalization"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
	Value float64
}

type ComputeU struct {
	Value bool
}

type Symmetric struct {
	Value bool
}

type InSitu struct {
	InitializeH bool
	InitializeU bool
	H           Matrix
	U           Matrix
	T1          Scalar
	T2          Scalar
	T3          Scalar
	S           Scalar
	// asymmetric case
	Hessenberg hessenbergReduction.InSitu
	Beta       Scalar
	Nu         Vector
	X          Vector
	T          Scalar
	T4         Vector
	// symmetric case
	C           Scalar
	Y           Scalar
	Z           Scalar
	Householder householderTridiagonalization.InSitu
}

/* -------------------------------------------------------------------------- */

func QRstep(H, U Matrix, p, q int, inSitu *InSitu) {

	var u Matrix

	m, _ := H.Dims()
	n := m - p - q

	H12 := H.Slice(0, p, p, m-q)
	H23 := H.Slice(p, m-q, m-q, m)
	H22 := H.Slice(p, m-q, p, m-q)

	if U != nil {
		u = U.Slice(0, m, p, m-q)
	}

	c := inSitu.S
	s := inSitu.T
	t1 := inSitu.T1
	t2 := inSitu.T2
	t3 := inSitu.T3

	// shift
	t3.Set(H22.At(n-1, n-1))
	for i := 0; i < n; i++ {
		g := H22.At(i, i)
		g.Sub(g, t3)
	}
	for i := 0; i < n-1; i++ {
		givensRotation.Run(H22.At(i, i), H22.At(i+1, i), c, s)
		// multiply with Givens matrix (G H)
		givensRotation.ApplyHessenbergLeft(H22, c, s, i, i+1, t1, t2)
		givensRotation.ApplyHessenbergLeft(H23, c, s, i, i+1, t1, t2)
		// multiply with Givens matrix (H G)
		givensRotation.ApplyHessenbergRight(H12, c, s, i, i+1, t1, t2)
		givensRotation.ApplyHessenbergRight(H22, c, s, i, i+1, t1, t2)
		if u != nil {
			givensRotation.ApplyRight(u, c, s, i, i+1, t1, t2)
		}
	}
	// shift
	for i := 0; i < n; i++ {
		g := H22.At(i, i)
		g.Add(g, t3)
	}
}

func francisQRstep(H, U Matrix, p, q int, inSitu *InSitu) {

	var u Matrix

	m, _ := H.Dims()
	n := m - p - q

	H12 := H.Slice(0, p, p, m-q)
	H23 := H.Slice(p, m-q, m-q, m)
	H22 := H.Slice(p, m-q, p, m-q)

	if U != nil {
		u = U.Slice(0, m, p, m-q)
	}

	s := inSitu.S
	t := inSitu.T
	x := inSitu.X
	t1 := inSitu.T1
	t2 := inSitu.T2
	t3 := inSitu.T3
	t4 := inSitu.T4

	beta := inSitu.Beta
	nu := inSitu.Nu

	h11 := H22.At(n-2, n-2)
	h12 := H22.At(n-2, n-1)
	h21 := H22.At(n-1, n-2)
	h22 := H22.At(n-1, n-1)

	s.Add(h11, h22)
	t1.Mul(h11, h22)
	t2.Mul(h12, h21)
	t.Sub(t1, t2)

	h11 = H22.At(0, 0)
	h12 = H22.At(0, 1)
	h21 = H22.At(1, 0)
	h22 = H22.At(1, 1)

	t1.Mul(h11, h11)
	t2.Mul(h12, h21)
	t3.Mul(s, h11)
	x.At(0).Add(t1, t2)
	x.At(0).Sub(x.At(0), t3)
	x.At(0).Add(x.At(0), t)

	x.At(1).Add(h11, h22)
	x.At(1).Sub(x.At(1), s)
	x.At(1).Mul(x.At(1), h21)

	x.At(2).Mul(h21, H22.At(2, 1))

	for k := 0; k < n-2; k++ {
		s := 1
		r := n
		if s < k {
			s = k
		}
		if r > k+4 {
			r = k + 4
		}
		householder.Run(x, beta, nu, t1, t2, t3)
		{
			h := H22.Slice(k, k+3, s-1, n)
			householder.ApplyLeft(h, beta, nu, t4.Slice(s-1, n), t1)
		}
		{
			h := H22.Slice(0, n, k, k+3)
			householder.ApplyRight(h, beta, nu, t4.Slice(0, n), t1)
		}
		{
			h := H12.Slice(0, p, k, k+3)
			householder.ApplyRight(h, beta, nu, t4.Slice(0, p), t1)
		}
		{
			h := H23.Slice(k, k+3, 0, q)
			householder.ApplyLeft(h, beta, nu, t4.Slice(0, q), t1)
		}
		if u != nil {
			u := u.Slice(0, m, k, k+3)
			householder.ApplyRight(u, beta, nu, t4.Slice(0, m), t1)
		}
		x.At(0).Set(H22.At(k+1, k))
		x.At(1).Set(H22.At(k+2, k))
		if k < n-3 {
			x.At(2).Set(H22.At(k+3, k))
		}
	}
	householder.Run(x.Slice(0, 2), beta, nu.Slice(0, 2), t1, t2, t3)
	{
		h := H22.Slice(n-2, n, n-3, n)
		householder.ApplyLeft(h, beta, nu.Slice(0, 2), t4.Slice(n-3, n), t1)
	}
	{
		h := H22.Slice(0, n, n-2, n)
		householder.ApplyRight(h, beta, nu.Slice(0, 2), t4.Slice(0, n), t1)
	}
	{
		h := H12.Slice(0, p, n-2, n)
		householder.ApplyRight(h, beta, nu.Slice(0, 2), t4.Slice(0, p), t1)
	}
	{
		h := H23.Slice(n-2, n, 0, q)
		householder.ApplyLeft(h, beta, nu.Slice(0, 2), t4.Slice(0, q), t1)
	}
	if u != nil {
		u := u.Slice(0, m, n-2, n)
		householder.ApplyRight(u, beta, nu.Slice(0, 2), t4.Slice(0, m), t1)
	}
}

/* -------------------------------------------------------------------------- */

func splitMatrix(h Matrix, q int) (int, int) {
	n, _ := h.Dims()
	// try increasing q
	for i := q; i < n-1; i++ {
		// k: start of the last block
		if h.At(n-i-1, n-i-2).GetValue() == 0.0 {
			q = i + 1
		}
		if i > q {
			break
		}
		if i == n-2 {
			q = i + 2
		}
	}
	p := n - q - 2
	if p < 0 {
		p = 0
	}
	// try decreasing p
	for p > 0 {
		if h.At(p, p-1).GetValue() != 0.0 {
			p -= 1
		} else {
			break
		}
	}
	return p, q
}

/* -------------------------------------------------------------------------- */

func qrAlgorithm(inSitu *InSitu, epsilon float64) (Matrix, Matrix, error) {

	h := inSitu.H
	u := inSitu.U
	n, _ := h.Dims()

	if h_, u_, err := hessenbergReduction.Run(h, &inSitu.Hessenberg, hessenbergReduction.ComputeU{u != nil}); err != nil {
		return nil, nil, err
	} else {
		h = h_
		u = u_
	}

	// apply Francis QR steps
	for p, q := 0, 0; q < n-1; {

		for i := 0; i < n-1; i++ {
			h11 := h.At(i, i).GetValue()
			h21 := h.At(i+1, i).GetValue()
			h22 := h.At(i+1, i+1).GetValue()
			if math.Abs(h21) <= epsilon*(math.Abs(h11)+math.Abs(h22)) {
				h.At(i+1, i).SetValue(0.0)
			}
		}
		// p: number of rows/cols in H11
		// q: number of rows/cols in H33
		p, q = splitMatrix(h, q)

		if q < n-1 {
			francisQRstep(h, u, p, q, inSitu)
		}
	}
	// reduce 2x2 blocks along the diagonal
	for i := 0; i < n-1; i++ {
		// check sub-diagonal element
		h21 := h.At(i+1, i).GetValue()
		if h21 == 0.0 {
			continue
		}
		// check if eigenvalues are complex
		h11 := h.At(i, i).GetValue()
		h12 := h.At(i, i+1).GetValue()
		h22 := h.At(i+1, i+1).GetValue()
		if (h11-h22)*(h11-h22)+4*h12*h21 < 0.0 {
			continue
		}
		// run QR steps until convergence
		for {
			h11 := h.At(i, i).GetValue()
			h21 := h.At(i+1, i).GetValue()
			h22 := h.At(i+1, i+1).GetValue()
			if math.Abs(h21) <= epsilon*(math.Abs(h11)+math.Abs(h22)) {
				h.At(i+1, i).SetValue(0.0)
				break
			} else {
				QRstep(h, u, i, n-i-2, inSitu)
			}
		}
	}
	return h, u, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

	n, m := a.Dims()
	t := a.ElementType()

	if n != m {
		return nil, nil, fmt.Errorf("`a' must be a square matrix")
	}
	inSitu := &InSitu{}
	epsilon := 1e-18
	computeU := false
	symmetric := false

	// loop over optional arguments
	for _, arg := range args {
		switch tmp := arg.(type) {
		case ComputeU:
			computeU = tmp.Value
		case Epsilon:
			epsilon = tmp.Value
		case Symmetric:
			symmetric = tmp.Value
		case *InSitu:
			inSitu = tmp
		case InSitu:
			panic("InSitu must be passed by reference")
		}
	}
	if inSitu.H == nil {
		inSitu.H = a.CloneMatrix()
		inSitu.Hessenberg.H = inSitu.H
	} else {
		if n1, m1 := inSitu.H.Dims(); n1 != n || m1 != m {
			return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
		}
		// initialize h if necessary
		if inSitu.H != a && inSitu.InitializeH {
			inSitu.H.Set(a)
		}
		inSitu.Hessenberg.H = inSitu.H
	}
	if computeU {
		if inSitu.U == nil {
			inSitu.U = NullMatrix(t, n, n)
			inSitu.Hessenberg.U = inSitu.U
		} else {
			if n1, m1 := inSitu.U.Dims(); n1 != n || m1 != m {
				return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
			}
			inSitu.Hessenberg.U = inSitu.U
		}
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
		inSitu.T4 = NullVector(t, m)
	}
	if inSitu.S == nil {
		inSitu.S = NullScalar(t)
	}
	if symmetric {
		if inSitu.C == nil {
			inSitu.C = NullScalar(t)
		}
		if inSitu.Y == nil {
			inSitu.Y = NullScalar(t)
		}
		if inSitu.Z == nil {
			inSitu.Z = NullScalar(t)
		}
		return qrAlgorithmSymmetric(inSitu, epsilon)
	} else {
		if inSitu.X == nil {
			inSitu.X = NullVector(t, 3)
		}
		if inSitu.Beta == nil {
			inSitu.Beta = NullScalar(t)
		}
		if inSitu.Nu == nil {
			inSitu.Nu = NullVector(t, 3)
		}
		if inSitu.T == nil {
			inSitu.T = NullScalar(t)
		}
		if inSitu.Hessenberg.T1 == nil {
			inSitu.Hessenberg.T1 = inSitu.T1
		}
		if inSitu.Hessenberg.T2 == nil {
			inSitu.Hessenberg.T2 = inSitu.T2
		}
		if inSitu.Hessenberg.T3 == nil {
			inSitu.Hessenberg.T3 = inSitu.T3
		}
		return qrAlgorithm(inSitu, epsilon)
	}
}
