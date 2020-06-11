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

package svd

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/householderBidiagonalization"
import "github.com/pbenner/autodiff/algorithm/givensRotation"

/* -------------------------------------------------------------------------- */

type ComputeU struct {
	Value bool
}

type ComputeV struct {
	Value bool
}

/* -------------------------------------------------------------------------- */

type Epsilon struct {
	Value float64
}

type InSitu struct {
	HouseholderBidiagonalization householderBidiagonalization.InSitu
	A                            Matrix
	U                            Matrix
	V                            Matrix
	Mu                           Scalar
	C                            Scalar
	S                            Scalar
	T1                           Scalar
	T2                           Scalar
	T3                           Scalar
	T4                           Scalar
	T5                           Scalar
}

/* -------------------------------------------------------------------------- */

// compute T = B^T B at rows/colums i,i+1
func computeSquare(t11, t12, t22 Scalar, B Matrix, i int) {
	b11 := B.At(i+0, i+0)
	b12 := B.At(i+0, i+1)
	b22 := B.At(i+1, i+1)
	t11.Mul(b11, b11) // b11^2
	t12.Mul(b12, b12) // b12^2
	t22.Mul(b22, b22) // b22^2
	t22.Add(t12, t22) // b12^2 + b22^2
	t12.Mul(b11, b12) // b11 b12
}

/* -------------------------------------------------------------------------- */

// compute the eigenvalue of a symmetric
// 2x2 matrix [ t11 t12; t12 t22 ] closer
// to t22
func wilkinsonShift(mu, t11, t12, t22, t1, t2 Scalar) {
	d := t1
	t := t2
	d.Sub(t11, t22)
	d.Div(d, ConstReal(2)) // d = (t11 - t22)/2

	t.Mul(t12, t12)
	mu.Mul(d, d)
	mu.Add(mu, t) // mu = d^2 + t12^2
	mu.Sqrt(mu)   // mu = sqrt(d^2 + t12^2)

	if d.GetValue() < 0.0 {
		mu.Neg(mu)
	}
	mu.Add(d, mu) // mu = d + sign(d) sqrt(d^2 + t12^2)
	mu.Div(t, mu) // mu = t12^2 / (d + sign(d) sqrt(d^2 + t12^2))

	mu.Sub(t22, mu) // mu = t22 - t12^2 / (d + sign(d) sqrt(d^2 + t12^2))
}

/* -------------------------------------------------------------------------- */

func golubKahanSVDstep(B, U, V Matrix, p int, inSitu *InSitu, epsilon float64) {

	_, n := B.Dims()

	mu := inSitu.Mu
	t11 := inSitu.T1
	t12 := inSitu.T2
	t22 := inSitu.T3
	t1 := inSitu.T4
	t2 := inSitu.T5

	computeSquare(t11, t12, t22, B, n-2)
	wilkinsonShift(mu, t11, t12, t22, t1, t2)
	computeSquare(t11, t12, t22, B, 0)

	y := t11
	y.Sub(y, mu)
	z := t12
	c := inSitu.C
	s := inSitu.S

	for k := 0; k < n-1; k++ {
		givensRotation.Run(y, z, c, s)
		givensRotation.ApplyBidiagRight(B, c, s, k, k+1, t1, t2)
		z.SetValue(0.0)
		if V != nil {
			givensRotation.ApplyRight(V, c, s, p+k, p+k+1, t1, t2)
		}
		y.Set(B.At(k+0, k))
		z.Set(B.At(k+1, k))
		givensRotation.Run(y, z, c, s)
		givensRotation.ApplyBidiagLeft(B, c, s, k, k+1, t1, t2)
		z.SetValue(0.0)
		if U != nil {
			givensRotation.ApplyLeft(U, c, s, p+k, p+k+1, t1, t2)
		}
		if k < n-2 {
			y.Set(B.At(k, k+1))
			z.Set(B.At(k, k+2))
		}
	}
}

/* -------------------------------------------------------------------------- */

func zeroRow(B, U, V Matrix, k int, inSitu *InSitu) {

	_, n := B.Dims()

	c := inSitu.C
	s := inSitu.S
	t1 := inSitu.T4
	t2 := inSitu.T5

	for i := k + 1; i < n; i++ {
		y := B.At(i, i)
		z := B.At(k, i)
		givensRotation.Run(y, z, c, s)
		givensRotation.ApplyBidiagLeft(B, c, s, i, k, t1, t2)
		if U != nil {
			givensRotation.ApplyLeft(U, c, s, i, k, t1, t2)
		}
		z.SetValue(0.0)
	}
}

/* -------------------------------------------------------------------------- */

func splitMatrix(B Matrix, q int) (int, int) {
	_, n := B.Dims()
	// try increasing q
	for q < n-1 {
		// fix a column
		k := n - q - 1
		// check if B33 is diagonal
		if B.At(k-1, k).GetValue() == 0.0 {
			q += 1
		} else {
			break
		}
	}
	if q == n-1 {
		q = n
	}
	p := n - q - 1
	// try decreasing p
	for p > 0 {
		k := p
		if B.At(k-1, k).GetValue() == 0.0 {
			break
		} else {
			p -= 1
		}
	}
	return p, q
}

/* -------------------------------------------------------------------------- */

func golubKahanSVD(inSitu *InSitu, epsilon float64) (Matrix, Matrix, Matrix, error) {

	A := inSitu.A

	_, n := A.Dims()

	computeU := householderBidiagonalization.ComputeU{inSitu.U != nil}
	computeV := householderBidiagonalization.ComputeV{inSitu.V != nil}

	H, U, V, _ := householderBidiagonalization.Run(A, computeU, computeV, &inSitu.HouseholderBidiagonalization)
	B := H.Slice(0, n, 0, n)

	for p, q := 0, 0; q < n; {

		for i := 0; i < n-1; i++ {
			b11 := B.At(i, i).GetValue()
			b12 := B.At(i, i+1).GetValue()
			b22 := B.At(i+1, i+1).GetValue()
			if math.Abs(b12) <= epsilon*(math.Abs(b11)+math.Abs(b22)) {
				B.At(i, i+1).SetValue(0.0)
			}
		}
		p, q = splitMatrix(B, q)

		if q < n-1 {
			// check diagonal elements in B22
			t := true
			for k := p; k < n-q-1; k++ {
				if B.At(k, k).GetValue() == 0.0 {
					zeroRow(B, U, V, k, inSitu)
					t = false
				}
			}
			if t {
				b := B.Slice(p, n-q, p, n-q)
				u := U
				v := V
				golubKahanSVDstep(b, u, v, p, inSitu, epsilon)
			}
		}
	}
	if U != nil {
		U = U.T()
	}
	return H, U, V, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, Matrix, error) {

	m, n := a.Dims()
	t := a.ElementType()

	if m < n {
		return nil, nil, nil, fmt.Errorf("`a' has invalid dimensions")
	}
	inSitu := &InSitu{}
	computeU := false
	computeV := false
	epsilon := 1.11e-16

	// loop over optional arguments
	for _, arg := range args {
		switch tmp := arg.(type) {
		case ComputeU:
			computeU = tmp.Value
		case ComputeV:
			computeV = tmp.Value
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
		// initialized by householderBidiagonalization
	} else {
		inSitu.U = nil
	}
	if computeV {
		if inSitu.V == nil {
			inSitu.V = NullMatrix(t, n, n)
		}
		// initialized by householderBidiagonalization
	} else {
		inSitu.V = nil
	}
	if inSitu.Mu == nil {
		inSitu.Mu = NullScalar(t)
	}
	if inSitu.C == nil {
		inSitu.C = NullScalar(t)
	}
	if inSitu.S == nil {
		inSitu.S = NullScalar(t)
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
		inSitu.T4 = NullScalar(t)
	}
	if inSitu.T5 == nil {
		inSitu.T5 = NullScalar(t)
	}
	// HouseholderBidiagonalization InSitu
	if inSitu.HouseholderBidiagonalization.A == nil {
		inSitu.HouseholderBidiagonalization.A = inSitu.A
	}
	if inSitu.HouseholderBidiagonalization.U == nil {
		inSitu.HouseholderBidiagonalization.U = inSitu.U
	}
	if inSitu.HouseholderBidiagonalization.V == nil {
		inSitu.HouseholderBidiagonalization.V = inSitu.V
	}
	if inSitu.HouseholderBidiagonalization.Beta == nil {
		inSitu.HouseholderBidiagonalization.Beta = inSitu.T4
	}
	if inSitu.HouseholderBidiagonalization.T1 == nil {
		inSitu.HouseholderBidiagonalization.T1 = inSitu.T1
	}
	if inSitu.HouseholderBidiagonalization.T2 == nil {
		inSitu.HouseholderBidiagonalization.T2 = inSitu.T2
	}
	if inSitu.HouseholderBidiagonalization.T3 == nil {
		inSitu.HouseholderBidiagonalization.T3 = inSitu.T2
	}
	return golubKahanSVD(inSitu, epsilon)
}
