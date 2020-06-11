/* Copyright (C) 2016 Philipp Benner
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
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type BasicEstimator interface {
	Estimate(gamma ConstVector, p ThreadPool) error
	GetParameters() Vector
	SetParameters(x Vector) error
	ScalarType() ScalarType
}

/* -------------------------------------------------------------------------- */

type ScalarEstimator interface {
	BasicEstimator
	CloneScalarEstimator() ScalarEstimator
	SetData(x ConstVector, n int) error
	GetEstimate() (ScalarPdf, error)
	EstimateOnData(x, gamma ConstVector, p ThreadPool) error
}

type VectorEstimator interface {
	BasicEstimator
	CloneVectorEstimator() VectorEstimator
	SetData(x []ConstVector, n int) error
	GetEstimate() (VectorPdf, error)
	Dim() int
	EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error
}

type MatrixEstimator interface {
	BasicEstimator
	CloneMatrixEstimator() MatrixEstimator
	SetData(x []ConstMatrix, n int) error
	GetEstimate() (MatrixPdf, error)
	Dims() (int, int)
	EstimateOnData(x []ConstMatrix, gamma ConstVector, p ThreadPool) error
}

/* -------------------------------------------------------------------------- */

type BasicBatchEstimator interface {
	Initialize(p ThreadPool) error
	GetParameters() Vector
	SetParameters(x Vector) error
	ScalarType() ScalarType
}

type ScalarBatchEstimator interface {
	BasicBatchEstimator
	CloneScalarBatchEstimator() ScalarBatchEstimator
	GetEstimate() (ScalarPdf, error)
	NewObservation(x ConstScalar, gamma ConstScalar, p ThreadPool) error
}

type VectorBatchEstimator interface {
	BasicBatchEstimator
	CloneVectorBatchEstimator() VectorBatchEstimator
	GetEstimate() (VectorPdf, error)
	Dim() int
	NewObservation(x ConstVector, gamma ConstScalar, p ThreadPool) error
}

type MatrixBatchEstimator interface {
	BasicBatchEstimator
	CloneMatrixBatchEstimator() MatrixBatchEstimator
	GetEstimate() (MatrixPdf, error)
	Dims() (int, int)
	NewObservation(x ConstMatrix, gamma ConstScalar, p ThreadPool) error
}
