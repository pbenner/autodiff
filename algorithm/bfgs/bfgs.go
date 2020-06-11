/* Copyright (C) 2016, 2017 Philipp Benner
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

// Reference:
// Nocedal, Jorge, and Stephen Wright. Numerical optimization.
// Springer Science & Business Media, 2006.

/* -------------------------------------------------------------------------- */

package bfgs

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/lineSearch"
import "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type Objective func(Vector) (Scalar, error)

type Hessian struct {
	Value Matrix
}

type Epsilon struct {
	Value float64
}

type MaxIterations struct {
	Value int
}

type Hook struct {
	Value func(x, gradient Vector, y Scalar) bool
}

type Constraints struct {
	Value func(x Vector) bool
}

/* -------------------------------------------------------------------------- */

type ObjectiveInSitu struct {
	Eval func(x, g Vector, y Scalar) error
	X    Vector
}

func newObjectiveInSitu(f Objective) ObjectiveInSitu {
	g := func(x, g Vector, y Scalar) error {
		z, err := f(x)
		if err != nil {
			return err
		}
		// copy value
		y.Set(z)
		// copy gradient
		for i := 0; i < z.GetN(); i++ {
			g.At(i).SetValue(z.GetDerivative(i))
		}
		return nil
	}
	return ObjectiveInSitu{g, nil}
}

func (f ObjectiveInSitu) Differentiate(x, g Vector, y Scalar) error {
	if f.X == nil {
		f.X = NullVector(RealType, x.Dim())
	}
	f.X.Set(x)
	f.X.Variables(1)
	if err := f.Eval(f.X, g, y); err != nil {
		return err
	}
	return nil
}

/* -------------------------------------------------------------------------- */

/* Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm:
 */

func bgfs_computeDirection(x Vector, y Scalar, g Vector, B Matrix, p Vector) {
	p.MdotV(B, g)
	for i := 0; i < x.Dim(); i++ {
		p.At(i).Neg(p.At(i))
	}
}

// update approximation of the Hessian matrix
func bfgs_updateB(g1, g2, p2 Vector, B1, B2 Matrix, t1, t2 Scalar, t3, t4 Vector, t5, t6 Matrix) bool {
	s := p2
	y := t3
	// y = Df(x2) - Df(x1)
	y.VsubV(g2, g1)
	// s.y
	t2.VdotV(s, y)
	// check if value is zero
	if math.Abs(t2.GetValue()) < 1e-16 {
		B2.Set(B1)
		return false
	}
	// y s^T
	t5.Outer(y, s)
	// B y s^T
	t5.MdotM(B1, t5)
	// s y^T B
	t6.Outer(s, y)
	// s y^T B
	t6.MdotM(t6, B1)
	// B y s^T + s y^T B
	t5.MaddM(t5, t6)
	// (B y s^T + s y^T B) / (s.y)
	t5.MdivS(t5, t2)
	// save result
	B2.MsubM(B1, t5)
	// y.B
	t4.VdotM(y, B1)
	// y.B.y
	t1.VdotV(t4, y)
	// s.y + y.B.y
	t1.Add(t1, t2)
	// (s.y)^2
	t2.Mul(t2, t2)
	// (s.y + y.B.y)/(s.y)^2
	t1.Div(t1, t2)
	// s s^T
	t5.Outer(s, s)
	// (s.y + y.B.y)/(s.y)^2 (s s^T)
	t5.MmulS(t5, t1)
	// save result
	B2.MaddM(B2, t5)
	return true
}

// update approximation of the inverse Hessian matrix
func bfgs_updateH(g1, g2, p2 Vector, H1, H2, I Matrix, t1, t2 Scalar, t3, t4 Vector, t5, t6 Matrix) bool {
	s := p2
	y := t3
	// y = Df(x2) - Df(x1)
	y.VsubV(g2, g1)
	// y^T s
	t1.VdotV(s, y)
	// check if value is zero
	if math.Abs(t1.GetValue()) == 0.0 {
		return false
	}
	// s y^T
	t5.Outer(s, y)
	// s y^T / (y^T s)
	t5.MdivS(t5, t1)
	// I - s y^T / (y^T s)
	t5.MsubM(I, t5)
	// [I - s y^T / (y^T s)] H1 [I - s y^T / (y^T s)]
	t6.MdotM(t5, H1)
	H2.MdotM(t6, t5)
	// s s^T
	t5.Outer(s, s)
	// s s^T / (y^T s)
	t5.MdivS(t5, t1)
	// [I - s y^T / (y^T s)] H1 [I - s y^T / (y^T s)] + s s^T / (y^T s)
	H2.MaddM(H2, t5)
	return true
}

func bfgs(f_ Objective, f ObjectiveInSitu, x0 Vector, H0 Matrix, epsilon Epsilon, maxIterations MaxIterations, hook Hook, constraints Constraints) (Vector, error) {

	// nomenclature:
	// B: Hessian
	// H: inverse Hessian
	n := x0.Dim()
	t := BareRealType

	p1 := NullVector(t, n)
	p2 := NullVector(t, n)
	P2 := NullVector(RealType, n)
	x1 := x0.CloneVector()
	x2 := x1.CloneVector()
	X2 := x1.CloneVector()
	y1 := NullScalar(t)
	y2 := NullScalar(t)
	g1 := NullVector(t, n)
	g2 := NullVector(t, n)
	H1 := H0.CloneMatrix()
	H2 := NullMatrix(t, n, n)
	// some temporary variables
	t1 := NullScalar(t)
	t2 := NullScalar(t)
	t3 := NullVector(t, n)
	t4 := NullVector(t, n)
	t5 := NullMatrix(t, n, n)
	t6 := NullMatrix(t, n, n)
	I := IdentityMatrix(t, n)

	equals := func(x1, x2 Vector) bool {
		for i := 0; i < x1.Dim(); i++ {
			if x1.At(i).GetValue() != x2.At(i).GetValue() {
				return false
			}
		}
		return true
	}
	// check initial value
	if constraints.Value != nil && !constraints.Value(x1) {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	// evaluate objective function
	if err := f.Differentiate(x1, g1, y1); err != nil {
		return x1, fmt.Errorf("invalid initial value: %s", err)
	}
	// evaluate stop criterion
	if t1.Vnorm(g1).GetValue() < epsilon.Value {
		return x1, nil
	}
	// execute hook if available
	if hook.Value != nil && hook.Value(x1, g1, y1) {
		return x1, nil
	}

	// keep track of whether H has been updated before
	first_update := true
	for i := 0; i < maxIterations.Value; i++ {
		bgfs_computeDirection(x1, y1, g1, H1, p1)
		// line search objective
		phi := func(alpha Scalar) (Scalar, error) {
			P2.VmulS(p1, alpha)
			X2.VaddV(x1, P2)
			return f_(X2)
		}
		// perform line search to find a new point x2
		alpha, err := lineSearch.Run(phi, BareRealType, lineSearch.Parameters{1, 100})
		// compute new position
		p2.VmulS(p1, alpha)
		x2.VaddV(x1, p2)

		if err != nil || equals(x1, x2) {
			// reset H to find a new direction
			if first_update {
				// the initial matrix H seems invalid, stop optimization here
				return x1, fmt.Errorf("line search failed")
			} else {
				first_update = true
				H2.Set(H0)
			}
		} else {
			// evaluate objective at new position
			if err := f.Differentiate(x2, g2, y2); err != nil {
				return x1, fmt.Errorf("invalid value: %s", err)
			}
			// execute hook if available
			if hook.Value != nil && hook.Value(x2, g2, y2) {
				break
			}
			// evaluate stop criterion
			if t1.Vnorm(g2).GetValue() < epsilon.Value {
				break
			}
			if first_update {
				// compute heuristic steplength y^T s / (y^T y)
				t3.VsubV(x2, x1)
				t4.VsubV(g2, g1)
				t1.VdotV(t3, t4)
				t2.VdotV(t4, t4)
				t1.Div(t1, t2)
				H1.MmulS(H1, t1)
			}
			// update inverse Hessian
			if ok := bfgs_updateH(g1, g2, p2, H1, H2, I, t1, t2, t3, t4, t5, t6); !ok {
				// reset H to find a new direction
				first_update = true
				H2.Set(H0)
			} else {
				first_update = false
			}
		}

		g1.Set(g2)
		x1.Set(x2)
		y1.Set(y2)
		p1.Set(p2)
		H1.Set(H2)
	}
	return x2, nil
}

/* -------------------------------------------------------------------------- */

// x0: starting point
// B0: initial approximation to the Hessian matrix

func Run(f Objective, x0 Vector, args ...interface{}) (Vector, error) {

	hessian := Hessian{nil}
	hook := Hook{nil}
	epsilon := Epsilon{1e-8}
	maxIterations := MaxIterations{int(^uint(0) >> 1)}
	constraints := Constraints{nil}

	n := x0.Dim()

	for _, arg := range args {
		switch a := arg.(type) {
		case Hessian:
			hessian = a
		case Hook:
			hook = a
		case Epsilon:
			epsilon = a
		case MaxIterations:
			maxIterations = a
		case Constraints:
			constraints = a
		default:
			panic("Bfgs(): Invalid optional argument!")
		}
	}
	if hessian.Value == nil {
		hessian.Value = IdentityMatrix(x0.ElementType(), n)
	} else {
		r, c := hessian.Value.Dims()
		if n != r || n != c {
			return nil, fmt.Errorf("argument dimensions do not match, i.e. x0 has length %d and B0 has dimension %dx%d\n", n, r, c)
		}
	}
	H, err := matrixInverse.Run(hessian.Value)
	if err != nil {
		return nil, err
	}
	return bfgs(f, newObjectiveInSitu(f), x0, H, epsilon, maxIterations, hook, constraints)
}
