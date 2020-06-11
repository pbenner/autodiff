/* Copyright (C) 2016-2017 Philipp Benner
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

package statistics

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type VectorClassifier interface {
	Eval(r Vector, x ConstVector) error
	Dim() int
	CloneVectorClassifier() VectorClassifier
}

type MatrixClassifier interface {
	Eval(r Vector, x ConstMatrix) error
	Dims() (int, int)
	CloneMatrixClassifier() MatrixClassifier
}

/* -------------------------------------------------------------------------- */

type ScalarBatchClassifier interface {
	Eval(r Scalar, x ConstScalar) error
	CloneScalarBatchClassifier() ScalarBatchClassifier
}

type VectorBatchClassifier interface {
	Eval(r Scalar, x ConstVector) error
	Dim() int
	CloneVectorBatchClassifier() VectorBatchClassifier
}

type MatrixBatchClassifier interface {
	Eval(r Scalar, x ConstMatrix) error
	Dims() (int, int)
	CloneMatrixBatchClassifier() MatrixBatchClassifier
}
