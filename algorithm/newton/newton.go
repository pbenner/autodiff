/* Copyright (C) 2015, 2017 Philipp Benner
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

package newton

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/matrixInverse"
import "github.com/pbenner/autodiff/algorithm/cholesky"
import "github.com/pbenner/autodiff/algorithm/lineSearch"
import "github.com/pbenner/autodiff/algorithm/qrAlgorithm"

/* -------------------------------------------------------------------------- */

type objective_root func(Vector) (Vector, Matrix, error)
type objective_crit func(Vector) (Vector, Matrix, error)
type objective_min func(Vector) (Scalar, Vector, Matrix, error)
type objective_line func(Scalar) (Scalar, error)

type Epsilon struct {
	Value float64
}

type HookRoot struct {
	Value func(Vector, Matrix, Vector) bool
}

type HookCrit HookRoot

type HookMin struct {
	Value func(Vector, Vector, Matrix, Scalar) bool
}

type HessianModification struct {
	Value string
}

type Constraints struct {
	Value func(x Vector) bool
}

type MaxIterations struct {
	Value int
}

type InSitu struct {
	T1       Vector
	T2       Scalar
	QR       qrAlgorithm.InSitu
	Cholesky cholesky.InSitu
	Inverse  matrixInverse.InSitu
}

/* -------------------------------------------------------------------------- */

func Vequals(x1, x2 Vector) bool {
	for i := 0; i < x1.Dim(); i++ {
		if x1.At(i).GetValue() != x2.At(i).GetValue() {
			return false
		}
	}
	return true
}

func getDirection(r, g Vector, H Matrix, hessianModification HessianModification, inSitu *InSitu) error {
	switch hessianModification.Value {
	case "Eigenvalue":
		delta := 1e-8
		inSitu.QR.InitializeH = true
		inSitu.QR.InitializeU = true
		h, u, _ := qrAlgorithm.Run(H, &inSitu.QR)
		for i := 0; i < g.Dim(); i++ {
			r := h.At(i, i)
			// elements on the diagonal are the eigenvalues, force them
			// to be positive
			if r.GetValue() < delta {
				r.SetValue(delta - r.GetValue())
			}
		}
		H.MdotM(u, h)
		H.MdotM(H, u.T())
		Q, err := matrixInverse.Run(H, &inSitu.Inverse, matrixInverse.PositiveDefinite{true})
		if err != nil {
			return err
		}
		r.MdotV(Q, g)
	case "LDL":
		// use modified cholesky LDL decomposition to obtain a valid search direction
		L, D, err := cholesky.Run(H, &inSitu.Cholesky, cholesky.LDL{true}, cholesky.ForcePD{true})
		if err != nil {
			return fmt.Errorf("inverting Hessian failed: %v", err)
		}
		// invert L
		Q, err := matrixInverse.Run(L.T(), &inSitu.Inverse, matrixInverse.UpperTriangular{true})
		if err != nil {
			return fmt.Errorf("inverting Hessian failed: %v", err)
		}
		c1 := NewBareReal(1.0)
		// invert D
		for i := 0; i < g.Dim(); i++ {
			D.At(i, i).Div(c1, D.At(i, i))
		}
		D.MdotM(Q, D)
		D.MdotM(D, Q.T())
		r.MdotV(D, g)
	case "None":
		// invert L
		Q, err := matrixInverse.Run(H, &inSitu.Inverse)
		if err != nil {
			return fmt.Errorf("inverting Hessian failed: %v", err)
		}
		r.MdotV(Q, g)
	default:
		panic(fmt.Sprintf("invalid hessian modification: %s", hessianModification.Value))
	}
	return nil
}

/* Newton's method for root finding
 * -------------------------------------------------------------------------- */

// nomenclature:
// f(x) = y
// J: Jacobian
func newton_root(f objective_root, x Vector,
	epsilon Epsilon,
	maxIterations MaxIterations,
	hook HookRoot,
	constraints Constraints,
	hessianModification HessianModification,
	inSitu *InSitu,
	options []interface{}) (Vector, error) {
	x1 := x.CloneVector()
	x2 := x.CloneVector()
	// variables for lineSearch
	c := NewBareReal(0.9)

	// check initial value
	if constraints.Value != nil && !constraints.Value(x1) {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	// evaluate objective function
	y, J, err := f(x1)
	if err != nil {
		return nil, err
	}
	// allocate temporary memory
	if inSitu.T1 == nil {
		inSitu.T1 = NullVector(y.ElementType(), y.Dim())
	}
	if inSitu.T2 == nil {
		inSitu.T2 = NullScalar(y.ElementType())
	}
	// temporary variables
	t1 := inSitu.T1
	t2 := inSitu.T2

	for i := 0; i < maxIterations.Value; i++ {
		// execute hook if available
		if hook.Value != nil && hook.Value(x1, J, y) {
			break
		}
		// evaluate stop criterion
		t2.Vnorm(y)
		if t2.GetValue() < epsilon.Value {
			break
		}
		if math.IsNaN(t2.GetValue()) {
			return x1, fmt.Errorf("NaN value detected")
		}
		if err := getDirection(t1, y, J, hessianModification, inSitu); err != nil {
			return nil, err
		}

		// this is a simplified line search that tries to
		// satisfy the constraints
		for {
			x2.VsubV(x1, t1)
			if Vequals(x1, x2) {
				return x1, fmt.Errorf("line search failed")
			}
			// check constraints
			if constraints.Value == nil || constraints.Value(x2) {
				// constraints are satisfied
				break
			}
			// decrease step size
			t1.VmulS(t1, c)
		}
		// evaluate objective function
		y, J, err = f(x2)
		if err != nil {
			return nil, err
		}
		// swap variables
		x1, x2 = x2, x1
	}
	return x1, nil
}

/* Newton's method for optimization
 * -------------------------------------------------------------------------- */

// nomenclature:
// f(x) = y
// g: gradient
// H: Hessian
func newton_min(
	f objective_min,
	x Vector,
	getPhi func(x, p Vector) objective_line,
	epsilon Epsilon,
	maxIterations MaxIterations,
	hook HookMin,
	constraints Constraints,
	hessianModification HessianModification,
	inSitu *InSitu,
	options []interface{}) (Vector, error) {
	x1 := x.CloneVector()
	x2 := x.CloneVector()
	// variables for lineSearch
	c := NewBareReal(0.9)
	var y1 Scalar
	var y2 Scalar

	// check initial value
	if constraints.Value != nil && !constraints.Value(x1) {
		return x1, fmt.Errorf("invalid initial value: %v", x1)
	}
	// evaluate objective function
	y, g, H, err := f(x1)
	if err != nil {
		return nil, err
	}
	if y != nil {
		y1 = y.CloneScalar()
		y2 = y.CloneScalar()
	}
	// allocate temporary memory
	if inSitu.T1 == nil {
		inSitu.T1 = NullVector(g.ElementType(), g.Dim())
	}
	if inSitu.T2 == nil {
		inSitu.T2 = NullScalar(g.ElementType())
	}
	// temporary variables
	t1 := inSitu.T1
	t2 := inSitu.T2
	// constraints function for the line search algorithm
	var constraints_line func(alpha Scalar) bool

	if constraints.Value != nil {
		constraints_line = func(alpha Scalar) bool {
			t1.VmulS(t1, alpha)
			x2.VsubV(x1, t1)
			return constraints.Value(x2)
		}
	}

	for i := 0; i < maxIterations.Value; i++ {
		// execute hook if available
		if hook.Value != nil && hook.Value(x1, g, H, y1) {
			break
		}
		// evaluate stop criterion
		t2.Vnorm(g)
		if t2.GetValue() < epsilon.Value {
			break
		}
		if math.IsNaN(t2.GetValue()) {
			return x1, fmt.Errorf("NaN value detected")
		}
		if err := getDirection(t1, g, H, hessianModification, inSitu); err != nil {
			return nil, err
		}

		if getPhi != nil {
			// get line search objective function
			phi := getPhi(x1, t1)
			// execute line search and update x
			if alpha, err := lineSearch.Run(phi, BareRealType,
				lineSearch.Constraints{constraints_line},
				lineSearch.Parameters{1, 20}); err != nil {
				return x1, err
			} else {
				t1.VmulS(t1, alpha)
				x2.VsubV(x1, t1)
			}
		} else {
			for {
				x2.VsubV(x1, t1)
				if Vequals(x1, x2) {
					return x1, fmt.Errorf("line search failed")
				}
				// check constraints
				if constraints.Value == nil || constraints.Value(x2) {
					break
				}
				// decrease step size
				t1.VmulS(t1, c)
			}
		}
		// evaluate objective function
		y, g, H, err = f(x2)
		if err != nil {
			return nil, err
		}
		y2.Set(y)
		x1, x2 = x2, x1
		y1, y2 = y2, y1
	}
	return x1, nil
}

/* -------------------------------------------------------------------------- */

func run_root(f objective_root, x Vector, args ...interface{}) (Vector, error) {

	hook := HookRoot{nil}
	epsilon := Epsilon{1e-8}
	constraints := Constraints{nil}
	hessianModification := HessianModification{"None"}
	maxIterations := MaxIterations{int(^uint(0) >> 1)}
	inSitu := &InSitu{}
	options := make([]interface{}, 0)

	for _, arg := range args {
		switch a := arg.(type) {
		case HookRoot:
			hook = a
		case HookCrit:
			hook = HookRoot(a)
		case Epsilon:
			epsilon = a
		case Constraints:
			constraints = a
		case HessianModification:
			hessianModification = a
		case MaxIterations:
			maxIterations = a
		case *InSitu:
			inSitu = a
		case InSitu:
			panic("InSitu must be passed by reference")
		default:
			options = append(options, a)
		}
	}

	return newton_root(f, x, epsilon, maxIterations, hook, constraints, hessianModification, inSitu, options)
}

func run_min(f objective_min, x Vector, getPhi func(x, p Vector) objective_line, args ...interface{}) (Vector, error) {

	hook := HookMin{nil}
	epsilon := Epsilon{1e-8}
	constraints := Constraints{nil}
	hessianModification := HessianModification{"None"}
	maxIterations := MaxIterations{int(^uint(0) >> 1)}
	inSitu := &InSitu{}
	options := make([]interface{}, 0)

	for _, arg := range args {
		switch a := arg.(type) {
		case HookMin:
			hook = a
		case Epsilon:
			epsilon = a
		case Constraints:
			constraints = a
		case HessianModification:
			hessianModification = a
		case MaxIterations:
			maxIterations = a
		case *InSitu:
			inSitu = a
		case InSitu:
			panic("InSitu must be passed by reference")
		default:
			options = append(options, a)
		}
	}

	return newton_min(f, x, getPhi, epsilon, maxIterations, hook, constraints, hessianModification, inSitu, options)
}

/* -------------------------------------------------------------------------- */

func RunRoot(f_ func(Vector) (Vector, error), x Vector, args ...interface{}) (Vector, error) {

	// Jacobian matrix
	var J Matrix
	var y Vector
	// copy of x for computing derivatives
	X := x.CloneVector()
	// objective function
	f := func(x Vector) (Vector, Matrix, error) {
		X.Set(x)
		if err := X.Variables(1); err != nil {
			return nil, nil, err
		}
		// evaluate objective function
		Y, err := f_(X)
		if err != nil {
			return nil, nil, err
		}
		if y == nil {
			y = NullVector(BareRealType, Y.Dim())
		}
		if J == nil {
			J = NullMatrix(BareRealType, y.Dim(), x.Dim())
		}
		// copy values to y
		for i := 0; i < y.Dim(); i++ {
			y.At(i).SetValue(Y.At(i).GetValue())
		}
		// copy derivatives to J
		for i := 0; i < y.Dim(); i++ {
			for j := 0; j < x.Dim(); j++ {
				J.At(i, j).SetValue(Y.At(i).GetDerivative(j))
			}
		}
		return y, J, nil
	}
	return run_root(f, x, args...)
}

func RunCrit(f_ func(Vector) (Scalar, error), x Vector, args ...interface{}) (Vector, error) {

	n := x.Dim()
	y := NullBareReal()
	g := NullVector(BareRealType, n)
	H := NullMatrix(BareRealType, n, n)
	// copy of x for computing derivatives
	X := x.CloneVector()
	// objective function
	f := func(x Vector) (Vector, Matrix, error) {
		X.Set(x)
		if err := X.Variables(2); err != nil {
			return nil, nil, err
		}
		// evaluate objective function
		Y, err := f_(X)
		if err != nil {
			return nil, nil, err
		}
		// copy function value to y
		y.SetValue(Y.GetValue())
		// copy derivatives to g
		for i := 0; i < n; i++ {
			g.At(i).SetValue(Y.GetDerivative(i))
		}
		// copy Hessian to H
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				H.At(i, j).SetValue(Y.GetHessian(i, j))
			}
		}
		return g, H, nil
	}
	return run_root(f, x, args...)
}

func RunMin(f_ func(Vector) (Scalar, error), x Vector, args ...interface{}) (Vector, error) {

	n := x.Dim()
	y := NullBareReal()
	g := NullVector(BareRealType, n)
	H := NullMatrix(BareRealType, n, n)
	// copy of x for computing derivatives
	X := x.CloneVector()
	P := x.CloneVector()
	// objective function
	f := func(x Vector) (Scalar, Vector, Matrix, error) {
		X.Set(x)
		if err := X.Variables(2); err != nil {
			return nil, nil, nil, err
		}
		// evaluate objective function
		Y, err := f_(X)
		if err != nil {
			return nil, nil, nil, err
		}
		// copy function value to y
		y.SetValue(Y.GetValue())
		// copy derivatives to g
		for i := 0; i < n; i++ {
			g.At(i).SetValue(Y.GetDerivative(i))
		}
		// copy Hessian to H
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				H.At(i, j).SetValue(Y.GetHessian(i, j))
			}
		}
		return y, g, H, nil
	}
	// objective function for line-search
	getPhi := func(x, p Vector) objective_line {
		phi := func(alpha Scalar) (Scalar, error) {
			P.VmulS(p, alpha)
			X.VsubV(x, P)
			return f_(X)
		}
		return phi
	}
	return run_min(f, x, getPhi, args...)
}
