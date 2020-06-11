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

package generic

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ProbabilityVector interface {
	Vector
	GetVector() Vector
	Normalize() error
	CloneProbabilityVector() ProbabilityVector
}

type HmmProbabilityVector struct {
	Vector
	t1 Scalar
	t2 Scalar
}

func NewHmmProbabilityVector(v Vector, isLog bool) (HmmProbabilityVector, error) {
	t1 := NewScalar(v.ElementType(), 0.0)
	t2 := NewScalar(v.ElementType(), 0.0)
	pi := v.CloneVector()
	// log-transform all probabilities
	if !isLog {
		pi.Map(func(x Scalar) { x.Log(x) })
	}
	r := HmmProbabilityVector{pi, t1, t2}
	if err := r.Normalize(); err != nil {
		return HmmProbabilityVector{}, nil
	}
	return r, nil
}

func (obj HmmProbabilityVector) GetVector() Vector {
	return obj.Vector
}

func (obj HmmProbabilityVector) Normalize() error {
	t1 := obj.t1
	t2 := obj.t2
	t1.SetValue(math.Inf(-1))
	for i := 0; i < obj.Dim(); i++ {
		t1.LogAdd(t1, obj.At(i), t2)
	}
	if math.IsInf(t1.GetValue(), -1) {
		return fmt.Errorf("normalization failed")
	}
	for i := 0; i < obj.Dim(); i++ {
		obj.At(i).Sub(obj.At(i), t1)
	}
	return nil
}

func (obj HmmProbabilityVector) CloneProbabilityVector() ProbabilityVector {
	return HmmProbabilityVector{
		obj.Vector.CloneVector(),
		obj.t1.CloneScalar(),
		obj.t2.CloneScalar()}
}

/* -------------------------------------------------------------------------- */

type TransitionMatrix interface {
	Matrix
	GetMatrix() Matrix
	Normalize() error
	CloneTransitionMatrix() TransitionMatrix
}

type HmmTransitionMatrix struct {
	Matrix
	t1 Scalar
	t2 Scalar
}

func NewHmmTransitionMatrix(tr_ Matrix, isLog bool) (HmmTransitionMatrix, error) {
	t1 := NewScalar(tr_.ElementType(), 0.0)
	t2 := NewScalar(tr_.ElementType(), 0.0)
	tr := tr_.CloneMatrix()
	// log-transform all probabilities
	if !isLog {
		tr.Map(func(x Scalar) { x.Log(x) })
	}
	r := HmmTransitionMatrix{tr, t1, t2}
	if err := r.Normalize(); err != nil {
		return HmmTransitionMatrix{}, err
	}
	return r, nil
}

func (obj HmmTransitionMatrix) GetMatrix() Matrix {
	return obj.Matrix
}

func (obj HmmTransitionMatrix) Normalize() error {
	t1 := obj.t1
	t2 := obj.t2
	n, m := obj.Dims()
	for i := 0; i < n; i++ {
		t1.SetValue(math.Inf(-1))
		for j := 0; j < m; j++ {
			t1.LogAdd(t1, obj.At(i, j), t2)
		}
		if math.IsInf(t1.GetValue(), -1) {
			obj.At(i, i).SetValue(0.0)
		} else {
			for j := 0; j < m; j++ {
				obj.At(i, j).Sub(obj.At(i, j), t1)
			}
		}
	}
	return nil
}

func (obj HmmTransitionMatrix) CloneTransitionMatrix() TransitionMatrix {
	return HmmTransitionMatrix{
		obj.Matrix.CloneMatrix(),
		obj.t1.CloneScalar(),
		obj.t2.CloneScalar()}
}
