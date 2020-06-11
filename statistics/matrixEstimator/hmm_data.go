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

type HmmDataSet interface {
	generic.HmmDataSet
	GetMappedData() []ConstVector
	EvaluateLogPdf(edist []VectorPdf, pool ThreadPool) error
}

/* -------------------------------------------------------------------------- */

type HmmStdDataRecord struct {
	offset int
	n      int
	p      Matrix
}

func (obj HmmStdDataRecord) MapIndex(k int) int {
	return obj.offset + k
}

func (obj HmmStdDataRecord) GetN() int {
	return obj.n
}

func (obj HmmStdDataRecord) LogPdf(r Scalar, c, k int) error {
	r.Set(obj.p.At(c, obj.MapIndex(k)))
	return nil
}

/* -------------------------------------------------------------------------- */

type HmmStdDataSet struct {
	// matrix of observations
	values  []ConstVector
	offsets []int
	// matrix with emission probabilities, each row corresponds
	// to an emission distribution and each column to a unique
	// observation
	p Matrix
	// number of observations
	n int
}

func NewHmmStdDataSet(t ScalarType, x []ConstMatrix, k int) (*HmmStdDataSet, error) {
	offsets := make([]int, len(x))
	values := []ConstVector{}
	n := 0              // number of data points
	_, m := x[0].Dims() // dimension of data points
	for d := 0; d < len(x); d++ {
		if xn, xm := x[d].Dims(); xm != m {
			return nil, fmt.Errorf("data has inconsistent dimensions")
		} else {
			for i := 0; i < xn; i++ {
				values = append(values, x[d].ConstRow(i))
			}
			offsets[d] = n
			n += xn
		}
	}
	r := HmmStdDataSet{}
	r.values = values
	r.offsets = offsets
	r.p = NullMatrix(t, k, len(values))
	r.n = n
	return &r, nil
}

func (obj *HmmStdDataSet) GetMappedData() []ConstVector {
	return obj.values
}

func (obj *HmmStdDataSet) GetRecord(i int) generic.HmmDataRecord {
	if i+1 == len(obj.offsets) {
		return HmmStdDataRecord{obj.offsets[i], len(obj.values) - obj.offsets[i], obj.p}
	} else {
		return HmmStdDataRecord{obj.offsets[i], obj.offsets[i+1] - obj.offsets[i], obj.p}
	}
}

func (obj *HmmStdDataSet) GetNMapped() int {
	return len(obj.values)
}

func (obj *HmmStdDataSet) GetNRecords() int {
	return len(obj.offsets)
}

func (obj *HmmStdDataSet) GetN() int {
	return obj.n
}

func (obj *HmmStdDataSet) EvaluateLogPdf(edist []VectorPdf, pool ThreadPool) error {
	x := obj.values
	p := obj.p
	m, n := obj.p.Dims()
	if len(edist) != m {
		return fmt.Errorf("data has invalid dimension")
	}
	// distributions may have state and must be cloned
	// for each thread
	d := make([][]VectorPdf, pool.NumberOfThreads())
	s := make([]float64, pool.NumberOfThreads())
	for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
		d[threadIdx] = make([]VectorPdf, m)
		for j := 0; j < m; j++ {
			d[threadIdx][j] = edist[j].CloneVectorPdf()
		}
	}
	g := pool.NewJobGroup()
	// evaluate emission distributions
	if err := pool.AddRangeJob(0, n, g, func(i int, pool ThreadPool, erf func() error) error {
		if erf() != nil {
			return nil
		}
		s[pool.GetThreadId()] = math.Inf(-1)
		// loop over emission distributions
		for j := 0; j < m; j++ {
			if err := d[pool.GetThreadId()][j].LogPdf(p.At(j, i), x[i]); err != nil {
				return err
			}
			s[pool.GetThreadId()] = LogAdd(s[pool.GetThreadId()], p.At(j, i).GetValue())
		}
		if math.IsInf(s[pool.GetThreadId()], -1) {
			return fmt.Errorf("probability is zero for all models on observation `%v'", x[i])
		}
		return nil
	}); err != nil {
		return fmt.Errorf("evaluating emission probabilities failed: %v", err)
	}
	if err := pool.Wait(g); err != nil {
		return fmt.Errorf("evaluating emission probabilities failed: %v", err)
	}
	return nil
}
