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

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"
import "github.com/pbenner/autodiff/statistics/matrixDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ShapeHmmEstimator struct {
	hmm1       *matrixDistribution.ShapeHmm
	hmm2       *matrixDistribution.ShapeHmm
	hmm3       *matrixDistribution.ShapeHmm
	data       *ShapeHmmDataSet
	estimators []*ShapeHmmAdapter
	// Baum-Welch arguments
	epsilon  float64
	maxSteps int
	args     []interface{}
}

/* -------------------------------------------------------------------------- */

func NewShapeHmmEstimator(pi Vector, tr Matrix, stateMap []int, batchEstimators []MatrixBatchEstimator, epsilon float64, maxSteps int, args ...interface{}) (*ShapeHmmEstimator, error) {
	if hmm, err := matrixDistribution.NewShapeHmm(pi, tr, stateMap, nil); err != nil {
		return nil, err
	} else {
		if hmm.NEDists() > 0 && len(batchEstimators) != hmm.NEDists() {
			return nil, fmt.Errorf("invalid number of estimators")
		}
		estimators := make([]*ShapeHmmAdapter, len(batchEstimators))
		for i, estimator := range batchEstimators {
			if r, err := NewShapeHmmAdapter(estimator); err != nil {
				return nil, err
			} else {
				estimators[i] = r
			}
		}
		for i, estimator := range estimators {
			// initialize distribution
			if hmm.Edist[i] == nil {
				if d, err := estimator.GetEstimate(); err != nil {
					return nil, err
				} else {
					hmm.Edist[i] = d
				}
			}
		}
		// initialize estimators with data
		r := ShapeHmmEstimator{}
		r.hmm1 = hmm.Clone()
		r.hmm2 = hmm.Clone()
		r.hmm3 = hmm.Clone()
		r.estimators = estimators
		r.epsilon = epsilon
		r.maxSteps = maxSteps
		r.args = args
		return &r, nil
	}
}

/* Baum-Welch interface
 * -------------------------------------------------------------------------- */

func (obj *ShapeHmmEstimator) GetBasicHmm() generic.BasicHmm {
	return obj.hmm1
}

func (obj *ShapeHmmEstimator) EvaluateLogPdf(pool ThreadPool) error {
	return obj.data.EvaluateLogPdf(obj.hmm2.Edist, pool)
}

func (obj *ShapeHmmEstimator) Swap() {
	obj.hmm1, obj.hmm2, obj.hmm3 = obj.hmm3, obj.hmm1, obj.hmm2
}

func (obj *ShapeHmmEstimator) Emissions(gamma []DenseBareRealVector, p ThreadPool) error {
	hmm1 := obj.hmm1
	hmm2 := obj.hmm2
	// estimate emission parameters
	g := p.NewJobGroup()
	if err := p.AddRangeJob(0, len(hmm1.Edist), g, func(c int, p ThreadPool, erf func() error) error {
		// copy parameters for faster convergence
		p1 := hmm1.Edist[c].GetParameters()
		p2 := hmm2.Edist[c].GetParameters()
		for j := 0; j < p1.Dim(); j++ {
			p1.At(j).Set(p2.At(j))
		}
		if err := obj.estimators[c].SetParameters(p1); err != nil {
			return err
		}
		// estimate parameters of the emission distribution
		if err := obj.estimators[c].Estimate(gamma[c], p); err != nil {
			return err
		}
		// update emission distribution
		if err := hmm1.Edist[c].SetParameters(obj.estimators[c].GetParameters()); err != nil {
			return err
		}
		return nil
	}); err != nil {
		return err
	}
	if err := p.Wait(g); err != nil {
		return err
	}
	return nil
}

func (obj *ShapeHmmEstimator) Step(meta ConstVector, tmp []generic.BaumWelchTmp, p ThreadPool) (float64, error) {
	hmm1 := obj.hmm1
	hmm2 := obj.hmm2
	return hmm1.Hmm.BaumWelchStep(&hmm1.Hmm, &hmm2.Hmm, obj.data, meta, tmp, p)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ShapeHmmEstimator) CloneMatrixEstimator() MatrixEstimator {
	estimators := make([]*ShapeHmmAdapter, len(obj.estimators))
	for i := 0; i < len(obj.estimators); i++ {
		estimators[i] = obj.estimators[i].Clone()
	}
	r := ShapeHmmEstimator{}
	r = *obj
	r.hmm1 = r.hmm1.Clone()
	r.hmm2 = r.hmm2.Clone()
	r.hmm3 = r.hmm3.Clone()
	r.estimators = estimators
	return &r
}

func (obj *ShapeHmmEstimator) Dims() (int, int) {
	return obj.hmm1.Dims()
}

func (obj *ShapeHmmEstimator) ScalarType() ScalarType {
	return obj.hmm1.ScalarType()
}

func (obj *ShapeHmmEstimator) GetParameters() Vector {
	return obj.hmm1.GetParameters()
}

func (obj *ShapeHmmEstimator) SetParameters(parameters Vector) error {
	return obj.hmm1.SetParameters(parameters)
}

func (obj *ShapeHmmEstimator) SetData(x []ConstMatrix, n int) error {
	if data, err := NewShapeHmmDataSet(obj.ScalarType(), x, obj.hmm1.NEDists()); err != nil {
		return err
	} else {
		for _, estimator := range obj.estimators {
			// set data
			if err := estimator.SetData(data.x, n); err != nil {
				return err
			}
		}
		obj.data = data
	}
	return nil
}

func (obj *ShapeHmmEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	data := obj.data
	nRecords := data.GetNRecords()
	nMapped := data.GetNMapped()
	nData := 0
	// determine length of the longest sequence
	for i := 0; i < data.GetNRecords(); i++ {
		r := data.GetRecord(i)
		if r.GetN() > nData {
			nData = r.GetN()
		}
	}
	return generic.BaumWelchAlgorithm(obj, gamma, nRecords, nData, nMapped, obj.hmm1.NStates(), obj.hmm1.NEDists(), obj.epsilon, obj.maxSteps, p, obj.args...)
}

func (obj *ShapeHmmEstimator) EstimateOnData(x []ConstMatrix, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *ShapeHmmEstimator) GetEstimate() (MatrixPdf, error) {
	return obj.hmm1, nil
}
