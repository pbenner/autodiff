/* Copyright (C) 2018 Philipp Benner
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

package generic

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/newton"

/* -------------------------------------------------------------------------- */

type EqualityConstraint [][2]int

func NewEqualityConstraint(x []int) (EqualityConstraint, error) {
	if len(x)%2 != 0 {
		return nil, fmt.Errorf("invalid constraints")
	}
	r := make(EqualityConstraint, len(x)/2)
	for i := 0; i < len(x); i += 2 {
		r[i/2][0] = x[i+0]
		r[i/2][1] = x[i+1]
	}
	return r, nil
}

/* -------------------------------------------------------------------------- */

type ChmmTransitionMatrix struct {
	Matrix
	constraints []EqualityConstraint
	counts      [][]int
}

func NewChmmTransitionMatrix(tr Matrix, constraints []EqualityConstraint, isLog bool) (ChmmTransitionMatrix, error) {
	return newChmmTransitionMatrix(tr, constraints, isLog, true)
}

func newChmmTransitionMatrix(tr Matrix, constraints []EqualityConstraint, isLog, normalize bool) (ChmmTransitionMatrix, error) {
	tr = tr.CloneMatrix()
	// log-transform all probabilities
	if !isLog {
		tr.Map(func(x Scalar) { x.Log(x) })
	}
	r := ChmmTransitionMatrix{tr, constraints, nil}
	if err := r.complementConstraints(); err != nil {
		return ChmmTransitionMatrix{}, err
	}
	r.computeCounts()

	if normalize {
		if err := r.Normalize(); err != nil {
			return ChmmTransitionMatrix{}, err
		}
	}
	return r, nil
}

func (obj ChmmTransitionMatrix) GetMatrix() Matrix {
	return obj.Matrix
}

func (obj ChmmTransitionMatrix) GetConstraints() []EqualityConstraint {
	return obj.constraints
}

func (obj ChmmTransitionMatrix) Normalize() error {
	t1 := NullBareReal()
	t2 := NullBareReal()
	n, m := obj.Dims()
	for i := 0; i < n; i++ {
		t1.SetValue(math.Inf(-1))
		for j := 0; j < m; j++ {
			t1.LogAdd(t1, obj.At(i, j), t2)
		}
		if math.IsInf(t1.GetValue(), -1) {
			obj.At(i, i).SetValue(0.0)
		}
	}
	if lambda, err := obj.computeLambda(); err != nil {
		return err
	} else {
		obj.normalize(lambda)
		return nil
	}
}

func (obj ChmmTransitionMatrix) CloneTransitionMatrix() TransitionMatrix {
	return ChmmTransitionMatrix{
		obj.Matrix.CloneMatrix(),
		obj.constraints,
		obj.counts}
}

/* -------------------------------------------------------------------------- */

func (obj *ChmmTransitionMatrix) complementConstraints() error {
	n, m := obj.Dims()
	cmap := make(map[[2]int]interface{})
	for _, constraint := range obj.constraints {
		for _, cell := range constraint {
			if _, ok := cmap[cell]; ok {
				return fmt.Errorf("entry (%d,%d) has multiple equality constraints", cell[0], cell[1])
			} else {
				cmap[cell] = nil
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if math.IsInf(obj.ConstAt(i, j).GetValue(), -1) {
				continue
			}
			if _, ok := cmap[[2]int{i, j}]; !ok {
				obj.constraints = append(obj.constraints, [][2]int{{i, j}})
			}
		}
	}
	return nil
}

func (obj *ChmmTransitionMatrix) computeCounts() {
	n, _ := obj.Dims()
	counts := make([][]int, len(obj.constraints))
	for i := 0; i < len(obj.constraints); i++ {
		counts[i] = make([]int, n)
		for j := 0; j < len(obj.constraints[i]); j++ {
			counts[i][obj.constraints[i][j][0]]++
		}
	}
	obj.counts = counts
}

/* -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) computeLambda() (Vector, error) {
	tr := obj.Matrix
	n, _ := tr.Dims()
	// lambda
	l := NullVector(RealType, n)
	x := NullVector(RealType, n)
	// objective function
	f := func(lambda Vector) (Vector, error) {
		obj.EvalConstraints(lambda, x)
		return x, nil
	}
	if r, err := newton.RunRoot(f, l, newton.Epsilon{1e-8}); err != nil {
		return nil, err
	} else {
		return r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) normalize(lambda ConstVector) {
	n, _ := obj.Dims()

	s := NewScalar(RealType, math.Inf(-1))
	t1 := NewScalar(RealType, math.Inf(-1))
	t2 := NewScalar(RealType, math.Inf(-1))
	// sum up xi
	sumXi := NullVector(BareRealType, len(obj.constraints))
	sumXi.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
	for k := 0; k < len(obj.constraints); k++ {
		for _, cell := range obj.constraints[k] {
			sumXi.At(k).LogAdd(sumXi.ConstAt(k), obj.ConstAt(cell[0], cell[1]), t2)
		}
	}
	// loop over constraints
	for k := 0; k < len(obj.constraints); k++ {
		// s = sum_j |sigma_q|_j lambda_j
		s.SetValue(math.Inf(-1))
		for j := 0; j < n; j++ {
			// counts[k][j]: number of times a variable appears in
			// row j constrained by k
			if obj.counts[k][j] == 0 {
				continue
			}
			t1.Add(ConstReal(math.Log(float64(obj.counts[k][j]))), lambda.ConstAt(j))
			s.LogAdd(s, t1, t2)
		}
		for _, cell := range obj.constraints[k] {
			obj.At(cell[0], cell[1]).Sub(sumXi.ConstAt(k), s)
		}
	}
}

/* evaluate transition matrix constraints given lagrangian multipliers
 * -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) evalConstraints(lambda ConstVector, x Vector) {
	n, _ := obj.Dims()

	s := NewScalar(RealType, math.Inf(-1))
	t1 := NewScalar(RealType, math.Inf(-1))
	t2 := NewScalar(RealType, math.Inf(-1))
	// reset x
	x.Map(func(xi Scalar) { xi.SetValue(math.Inf(-1)) })
	// sum up xi
	sumXi := NullVector(BareRealType, len(obj.constraints))
	sumXi.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
	for k := 0; k < len(obj.constraints); k++ {
		for _, cell := range obj.constraints[k] {
			sumXi.At(k).LogAdd(sumXi.ConstAt(k), obj.ConstAt(cell[0], cell[1]), t2)
		}
	}
	// loop over rows
	for i := 0; i < n; i++ {
		// loop over constraints
		for k := 0; k < len(obj.constraints); k++ {
			// check if at row i there is a variable constrained
			// by k
			if obj.counts[k][i] == 0 {
				continue
			}
			// s = sum_j |sigma_q|_j lambda_j
			s.SetValue(math.Inf(-1))
			for j := 0; j < n; j++ {
				// counts[k][j]: number of times a variable appears in
				// row j constrained by k
				if obj.counts[k][j] == 0 {
					continue
				}
				t1.Add(ConstReal(math.Log(float64(obj.counts[k][j]))), lambda.ConstAt(j))
				s.LogAdd(s, t1, t2)
			}
			// t1 = |sigma_q|_i xi_q
			t1.Add(ConstReal(math.Log(float64(obj.counts[k][i]))), sumXi.ConstAt(k))
			// t1 = |sigma_q|_i xi_q / (sum_j |sigma_q|_j lambda_j)
			t1.Sub(t1, s)
			// add result to x
			x.At(i).LogAdd(x.At(i), t1, t2)
		}
	}
}

func (obj ChmmTransitionMatrix) EvalConstraints(lambda ConstVector, x Vector) error {
	tr := obj.Matrix
	n, m := tr.Dims()
	if n != m {
		return fmt.Errorf("tr is not a square-matrix")
	}
	if n != lambda.Dim() {
		return fmt.Errorf("lambda has invalid dimension")
	}
	if n != x.Dim() {
		return fmt.Errorf("x has invalid dimension")
	}
	obj.evalConstraints(lambda, x)

	x.Map(func(xi Scalar) {
		xi.Exp(xi)
		xi.Sub(xi, ConstReal(1.0))
	})

	return nil
}
