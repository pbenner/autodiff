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

package vectorEstimator

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
	GetMappedData() ConstVector
	EvaluateLogPdf(edist []ScalarPdf, pool ThreadPool) error
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
	// vector of observations
	values  Vector
	offsets []int
	// matrix with emission probabilities, each row corresponds
	// to an emission distribution and each column to a unique
	// observation
	p Matrix
	// number of observations
	n int
}

func NewHmmStdDataSet(t ScalarType, x []ConstVector, k int) (*HmmStdDataSet, error) {
	offsets := make([]int, len(x))
	n := 0
	for d := 0; d < len(x); d++ {
		offsets[d] = n
		n += x[d].Dim()
	}
	values := NullVector(x[0].ElementType(), n)
	for d := 0; d < len(x); d++ {
		values.Slice(offsets[d], offsets[d]+x[d].Dim()).Set(x[d])
	}
	r := HmmStdDataSet{}
	r.values = values
	r.offsets = offsets
	r.p = NullMatrix(t, k, values.Dim())
	r.n = n
	return &r, nil
}

func (obj *HmmStdDataSet) GetMappedData() ConstVector {
	return obj.values
}

func (obj *HmmStdDataSet) GetRecord(i int) generic.HmmDataRecord {
	if i+1 == len(obj.offsets) {
		return HmmStdDataRecord{obj.offsets[i], obj.values.Dim() - obj.offsets[i], obj.p}
	} else {
		return HmmStdDataRecord{obj.offsets[i], obj.offsets[i+1] - obj.offsets[i], obj.p}
	}
}

func (obj *HmmStdDataSet) GetNMapped() int {
	return obj.values.Dim()
}

func (obj *HmmStdDataSet) GetNRecords() int {
	return len(obj.offsets)
}

func (obj *HmmStdDataSet) GetN() int {
	return obj.n
}

func (obj *HmmStdDataSet) EvaluateLogPdf(edist []ScalarPdf, pool ThreadPool) error {
	x := obj.values
	p := obj.p
	m, n := obj.p.Dims()
	if len(edist) != m {
		return fmt.Errorf("data has invalid dimension")
	}
	// distributions may have state and must be cloned
	// for each thread
	d := make([][]ScalarPdf, pool.NumberOfThreads())
	s := make([]float64, pool.NumberOfThreads())
	for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
		d[threadIdx] = make([]ScalarPdf, m)
		for j := 0; j < m; j++ {
			d[threadIdx][j] = edist[j].CloneScalarPdf()
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
			if err := d[pool.GetThreadId()][j].LogPdf(p.At(j, i), x.At(i)); err != nil {
				return err
			}
			s[pool.GetThreadId()] = LogAdd(s[pool.GetThreadId()], p.At(j, i).GetValue())
		}
		if math.IsInf(s[pool.GetThreadId()], -1) {
			return fmt.Errorf("probability is zero for all models on observation `%v'", x.At(i))
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

/* -------------------------------------------------------------------------- */

type HmmSummarizedDataRecord struct {
	index []int
	p     Matrix
}

func (obj HmmSummarizedDataRecord) MapIndex(k int) int {
	return obj.index[k]
}

func (obj HmmSummarizedDataRecord) GetN() int {
	return len(obj.index)
}

func (obj HmmSummarizedDataRecord) LogPdf(r Scalar, c, k int) error {
	r.Set(obj.p.At(c, obj.MapIndex(k)))
	return nil
}

/* -------------------------------------------------------------------------- */

type HmmSummarizedDataSet struct {
	// vector of unique observations
	values Vector
	index  [][]int
	// matrix with emission probabilities, each row corresponds
	// to an emission distribution and each column to a unique
	// observation
	p Matrix
	// number of observations
	n int
}

func NewHmmSummarizedDataSet(t ScalarType, x []Vector, k int) (*HmmSummarizedDataSet, error) {
	xMap := make(map[[1]float64]int)
	index := make([][]int, len(x))
	values := NullVector(x[0].ElementType(), 0)
	m := 0
	// convert vector elements to arrays, which can be used
	// as keys for xMap
	datum := [1]float64{0}
	for d := 0; d < len(x); d++ {
		index[d] = make([]int, x[d].Dim())
		for i := 0; i < x[d].Dim(); i++ {
			datum[0] = x[d].At(i).GetValue()
			if idx, ok := xMap[datum]; ok {
				index[d][i] = idx
			} else {
				idx := values.Dim()
				values = values.AppendScalar(x[d].At(i))
				xMap[datum] = idx
				index[d][i] = idx
			}
		}
		m += x[d].Dim()
	}
	r := HmmSummarizedDataSet{}
	r.values = values
	r.index = index
	r.p = NullMatrix(t, k, values.Dim())
	r.n = m
	return &r, nil
}

func (obj *HmmSummarizedDataSet) GetMappedData() ConstVector {
	return obj.values
}

func (obj *HmmSummarizedDataSet) GetRecord(i int) generic.HmmDataRecord {
	return HmmSummarizedDataRecord{obj.index[i], obj.p}
}

func (obj *HmmSummarizedDataSet) GetNMapped() int {
	return obj.values.Dim()
}

func (obj *HmmSummarizedDataSet) GetNRecords() int {
	return len(obj.index)
}

func (obj *HmmSummarizedDataSet) GetN() int {
	return obj.n
}

func (obj *HmmSummarizedDataSet) EvaluateLogPdf(edist []ScalarPdf, pool ThreadPool) error {
	x := obj.values
	p := obj.p
	m, n := obj.p.Dims()
	if len(edist) != m {
		return fmt.Errorf("data has invalid dimension")
	}
	// distributions may have state and must be cloned
	// for each thread
	d := make([][]ScalarPdf, pool.NumberOfThreads())
	s := make([]float64, pool.NumberOfThreads())
	for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
		d[threadIdx] = make([]ScalarPdf, m)
		for j := 0; j < m; j++ {
			d[threadIdx][j] = edist[j].CloneScalarPdf()
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
			if err := d[pool.GetThreadId()][j].LogPdf(p.At(j, i), x.At(i)); err != nil {
				return err
			}
			s[pool.GetThreadId()] = LogAdd(s[pool.GetThreadId()], p.At(j, i).GetValue())
		}
		if math.IsInf(s[pool.GetThreadId()], -1) {
			return fmt.Errorf("probability is zero for all models on observation `%v'", x.At(i))
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
