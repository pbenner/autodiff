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

package matrixInverse

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/cholesky"
import "github.com/pbenner/autodiff/algorithm/gaussJordan"
import "github.com/pbenner/autodiff/algorithm/rprop"

/* -------------------------------------------------------------------------- */

type PositiveDefinite struct {
	Value bool
}

type UpperTriangular struct {
	Value bool
}

type InSitu struct {
	Id       Matrix
	A        Matrix
	B        Vector
	Cholesky cholesky.InSitu
}

/* -------------------------------------------------------------------------- */

// compute the inverse of a matrix with a
// gradient descent method
func mInverseGradient(matrix ConstMatrix) (Matrix, error) {
	rows, cols := matrix.Dims()
	if rows != cols {
		panic("MInverse(): Not a square matrix!")
	}
	I := IdentityMatrix(matrix.ElementType(), rows)
	r := NullMatrix(BareRealType, rows, cols)
	r.Set(matrix)
	s := NewScalar(matrix.ElementType(), 0.0)
	// objective function
	f := func(x Vector) (Scalar, error) {
		r.AsVector().Set(x)
		s.Mnorm(r.MsubM(r.MdotM(matrix, r), I))
		return s, nil
	}
	x, _ := rprop.Run(f, r.AsVector(), 0.01, []float64{2.0, 0.1})
	r.AsVector().Set(x)
	return r, nil
}

func mInverse(matrix ConstMatrix, inSitu *InSitu, args ...interface{}) (Matrix, error) {
	a := inSitu.A
	x := inSitu.Id
	b := inSitu.B
	// copy values to a
	a.Set(matrix)
	// initialize b with ones
	for i := 0; i < b.Dim(); i++ {
		b.At(i).SetValue(1.0)
	}
	// call Gauss-Jordan algorithm
	if err := gaussJordan.Run(a, x, b, args...); err != nil {
		return nil, err
	} else {
		return x, nil
	}
}

func mInversePositiveDefinite(matrix ConstMatrix, inSitu *InSitu, args ...interface{}) (Matrix, error) {
	a, _, err := cholesky.Run(matrix, &inSitu.Cholesky)
	if err != nil {
		return nil, err
	}
	a = a.T()
	x := inSitu.Id
	b := inSitu.B
	// initialize b with ones
	for i := 0; i < b.Dim(); i++ {
		b.At(i).SetValue(1.0)
	}
	args = append(args, gaussJordan.UpperTriangular{true})
	// call Gauss-Jordan algorithm
	if err := gaussJordan.Run(a, x, b, args...); err != nil {
		return nil, err
	} else {
		// recycle a to store the result
		return a.MdotM(x, x.T()), nil
	}
}

func mInverseUpperTriangular(matrix ConstMatrix, inSitu *InSitu, args ...interface{}) (Matrix, error) {
	a := inSitu.A
	x := inSitu.Id
	b := inSitu.B
	// copy values to a
	a.Set(matrix)
	// initialize b with ones
	for i := 0; i < b.Dim(); i++ {
		b.At(i).SetValue(1.0)
	}
	args = append(args, gaussJordan.UpperTriangular{true})
	// call Gauss-Jordan algorithm
	if err := gaussJordan.Run(a, x, b, args...); err != nil {
		return nil, err
	} else {
		return x, nil
	}
}

/* -------------------------------------------------------------------------- */

func Run(matrix ConstMatrix, args ...interface{}) (Matrix, error) {
	rows, cols := matrix.Dims()
	if rows != cols {
		panic("not a square matrix")
	}
	if rows == 0 {
		panic("empty matrix")
	}
	positiveDefinite := false
	upperTriangular := false
	inSitu := &InSitu{}

	gArgs := []interface{}{}

	// loop over optional arguments
	for _, arg := range args {
		switch a := arg.(type) {
		case PositiveDefinite:
			positiveDefinite = a.Value
		case UpperTriangular:
			upperTriangular = a.Value
		case *InSitu:
			inSitu = a
		case InSitu:
			panic("InSitu must be passed by reference")
		default:
			// all other arguments are passed to the
			// Gauss-Jordan algorithm
			gArgs = append(gArgs, arg)
		}
	}
	if inSitu.Id == nil {
		inSitu.Id = IdentityMatrix(matrix.ElementType(), rows)
	} else {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if i == j {
					inSitu.Id.At(i, j).SetValue(1.0)
				} else {
					inSitu.Id.At(i, j).SetValue(0.0)
				}
			}
		}
	}
	if inSitu.A == nil && positiveDefinite == false {
		inSitu.A = NullMatrix(matrix.ElementType(), rows, rows)
	}
	if inSitu.B == nil {
		inSitu.B = NullVector(matrix.ElementType(), rows)
	}
	if positiveDefinite {
		return mInversePositiveDefinite(matrix, inSitu, gArgs...)
	} else {
		if upperTriangular {
			return mInverseUpperTriangular(matrix, inSitu, gArgs...)
		} else {
			return mInverse(matrix, inSitu, gArgs...)
		}
	}
}
