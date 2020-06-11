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

package matrixEstimator

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ShapeHmmDataSet struct {
	// vector of unique observations
	x       []ConstMatrix
	offsets []int
	// matrix with emission probabilities, each row corresponds
	// to an emission distribution and each column to a unique
	// observation
	p Matrix
	// number of observations
	n int
}

func NewShapeHmmDataSet(t ScalarType, x []ConstMatrix, k int) (*ShapeHmmDataSet, error) {
	offsets := make([]int, len(x))
	if len(x) == 0 {
		return &ShapeHmmDataSet{}, nil
	}
	n := 0              // number of data points
	_, m := x[0].Dims() // dimension of data points
	for d := 0; d < len(x); d++ {
		if xn, xm := x[d].Dims(); xm != m {
			return nil, fmt.Errorf("data has inconsistent dimensions")
		} else {
			offsets[d] = n
			n += xn
		}
	}
	r := ShapeHmmDataSet{}
	r.offsets = offsets
	r.x = x
	r.p = NullMatrix(t, k, n)
	r.n = n
	return &r, nil
}

func (obj *ShapeHmmDataSet) GetRecord(i int) generic.HmmDataRecord {
	if i+1 == len(obj.offsets) {
		return HmmStdDataRecord{obj.offsets[i], obj.n - obj.offsets[i], obj.p}
	} else {
		return HmmStdDataRecord{obj.offsets[i], obj.offsets[i+1] - obj.offsets[i], obj.p}
	}
}

func (obj *ShapeHmmDataSet) GetNMapped() int {
	return obj.n
}

func (obj *ShapeHmmDataSet) GetNRecords() int {
	return len(obj.x)
}

func (obj *ShapeHmmDataSet) GetN() int {
	return obj.n
}

func (obj *ShapeHmmDataSet) EvaluateLogPdf(edist []MatrixPdf, pool ThreadPool) error {
	r, c := edist[0].Dims()
	p := obj.p
	m, _ := obj.p.Dims()
	if len(edist) != m {
		return fmt.Errorf("data has invalid dimension")
	}
	g := pool.NewJobGroup()
	// distributions may have state and must be cloned
	// for each thread
	d := make([][]MatrixPdf, pool.NumberOfThreads())
	s := make([]float64, pool.NumberOfThreads())
	// clone distributions to ensure thread safety
	for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
		d[threadIdx] = make([]MatrixPdf, m)
		for j := 0; j < m; j++ {
			d[threadIdx][j] = edist[j].CloneMatrixPdf()
		}
	}
	// evaluate distributions
	for iRecord := 0; iRecord < obj.GetNRecords(); iRecord++ {
		record := obj.GetRecord(iRecord)
		x := obj.x[iRecord]
		n, xm := x.Dims()
		if xm != c {
			return fmt.Errorf("data has invalid dimension")
		}
		pool.AddRangeJob(0, n-r, g, func(i int, pool ThreadPool, erf func() error) error {
			if erf() != nil {
				return nil
			}
			k := record.MapIndex(i + r/2)
			d := d[pool.GetThreadId()]
			s := s[pool.GetThreadId()]
			s = math.Inf(-1)
			x := x.ConstSlice(i, i+r, 0, c)
			// loop over distributions
			for j := 0; j < m; j++ {
				if err := d[j].LogPdf(p.At(j, k), x); err != nil {
					return err
				}
				s = LogAdd(s, p.At(j, k).GetValue())
			}
			if math.IsInf(s, -1) {
				return fmt.Errorf("probability is zero for all models on observation `%v'", x)
			}
			return nil
		})
	}
	if err := pool.Wait(g); err != nil {
		return fmt.Errorf("evaluating emission probabilities failed: %v", err)
	}
	return nil
}
