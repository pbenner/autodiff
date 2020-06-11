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
import "os"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type HmmEstimator struct {
	hmm1       *vectorDistribution.Hmm
	hmm2       *vectorDistribution.Hmm
	hmm3       *vectorDistribution.Hmm
	data       HmmDataSet
	estimators []ScalarEstimator
	// Baum-Welch arguments
	epsilon  float64
	maxSteps int
	args     []interface{}
	// split data into smaller pieces
	// (disabled if set to 0)
	ChunkSize int
	// hook options
	SaveFile     string
	SaveInterval int
	Trace        string
	Verbose      int
	// estimator options
	OptimizeEmissions   bool
	OptimizeTransitions bool
}

/* -------------------------------------------------------------------------- */

func NewHmmEstimator(pi Vector, tr Matrix, stateMap, startStates, finalStates []int, estimators []ScalarEstimator, epsilon float64, maxSteps int, args ...interface{}) (*HmmEstimator, error) {
	if hmm, err := vectorDistribution.NewHmm(pi, tr, stateMap, nil); err != nil {
		return nil, err
	} else {
		if err := hmm.SetStartStates(startStates); err != nil {
			return nil, err
		}
		if err := hmm.SetFinalStates(finalStates); err != nil {
			return nil, err
		}
		if hmm.NEDists() > 0 && len(estimators) != hmm.NEDists() {
			return nil, fmt.Errorf("invalid number of estimators")
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
		r := HmmEstimator{}
		r.hmm1 = hmm.Clone()
		r.hmm2 = hmm.Clone()
		r.hmm3 = hmm.Clone()
		r.estimators = estimators
		r.epsilon = epsilon
		r.maxSteps = maxSteps
		r.args = args
		r.OptimizeEmissions = true
		r.OptimizeTransitions = true
		return &r, nil
	}
}

/* Baum-Welch interface
 * -------------------------------------------------------------------------- */

func (obj *HmmEstimator) GetBasicHmm() generic.BasicHmm {
	return obj.hmm1
}

func (obj *HmmEstimator) EvaluateLogPdf(pool ThreadPool) error {
	return obj.data.EvaluateLogPdf(obj.hmm2.Edist, pool)
}

func (obj *HmmEstimator) Swap() {
	obj.hmm1, obj.hmm2, obj.hmm3 = obj.hmm3, obj.hmm1, obj.hmm2
}

func (obj *HmmEstimator) Emissions(gamma []DenseBareRealVector, p ThreadPool) error {
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

func (obj *HmmEstimator) Step(meta ConstVector, tmp []generic.BaumWelchTmp, p ThreadPool) (float64, error) {
	hmm1 := obj.hmm1
	hmm2 := obj.hmm2
	return hmm1.Hmm.BaumWelchStep(&hmm1.Hmm, &hmm2.Hmm, obj.data, meta, tmp, p)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *HmmEstimator) CloneVectorEstimator() VectorEstimator {
	estimators := make([]ScalarEstimator, len(obj.estimators))
	for i := 0; i < len(obj.estimators); i++ {
		estimators[i] = obj.estimators[i].CloneScalarEstimator()
	}
	r := HmmEstimator{}
	r = *obj
	r.hmm1 = r.hmm1.Clone()
	r.hmm2 = r.hmm2.Clone()
	r.hmm3 = r.hmm3.Clone()
	r.estimators = estimators
	return &r
}

func (obj *HmmEstimator) Dim() int {
	return obj.hmm1.Dim()
}

func (obj *HmmEstimator) ScalarType() ScalarType {
	return obj.hmm1.ScalarType()
}

func (obj *HmmEstimator) GetParameters() Vector {
	return obj.hmm1.GetParameters()
}

func (obj *HmmEstimator) SetParameters(parameters Vector) error {
	return obj.hmm1.SetParameters(parameters)
}

func (obj *HmmEstimator) SetData(x []ConstVector, n int) error {
	// split data into chunks
	//////////////////////////////////////////////////////////////////////////////
	if obj.ChunkSize > 0 {
		var x_ []ConstVector
		for i := 0; i < len(x); i++ {
			m := x[i].Dim()
			for j := 0; j < m; j += obj.ChunkSize {
				jFrom := j
				jTo := j + obj.ChunkSize
				if jTo > m {
					jTo = m
				}
				x_ = append(x_, x[i].ConstSlice(jFrom, jTo))
			}
		}
		x = x_
	}
	if data, err := NewHmmStdDataSet(obj.ScalarType(), x, obj.hmm1.NEDists()); err != nil {
		return err
	} else {
		for i, estimator := range obj.estimators {
			// set data
			if err := estimator.SetData(data.GetMappedData(), n); err != nil {
				return err
			}
			// initialize distribution
			if d, err := estimator.GetEstimate(); err != nil {
				return err
			} else {
				obj.hmm1.Edist[i] = d.CloneScalarPdf()
				obj.hmm2.Edist[i] = d.CloneScalarPdf()
				obj.hmm3.Edist[i] = d.CloneScalarPdf()
			}
		}
		obj.data = data
	}
	return nil
}

func (obj *HmmEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	hook_save := generic.BaumWelchHook{}
	hook_trace := generic.BaumWelchHook{}
	hook_verbose := generic.BaumWelchHook{}
	trace := NullVector(obj.ScalarType(), 0)
	if obj.SaveFile != "" && obj.SaveInterval > 0 {
		hook_save.Value = func(hmm generic.BasicHmm, i int, likelihood, epsilon float64) {
			if i%obj.SaveInterval == 0 {
				if d, err := obj.GetEstimate(); err == nil {
					ExportDistribution(obj.SaveFile, d)
				}
			}
		}
	}
	// add hooks
	//////////////////////////////////////////////////////////////////////////////
	if obj.Trace != "" {
		hook_trace.Value = func(hmm generic.BasicHmm, i int, likelihood, epsilon float64) {
			trace = trace.AppendVector(hmm.GetParameters())
		}
	}
	if obj.Verbose > 1 {
		hook_verbose = generic.DefaultBaumWelchHook(os.Stderr)
	} else if obj.Verbose > 0 {
		hook_verbose = generic.PlainBaumWelchHook(os.Stderr)
	}
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
	args := obj.args
	args = append(args, hook_save)
	args = append(args, hook_trace)
	args = append(args, hook_verbose)
	args = append(args, generic.BaumWelchOptimizeEmissions{obj.OptimizeEmissions})
	args = append(args, generic.BaumWelchOptimizeTransitions{obj.OptimizeTransitions})
	return generic.BaumWelchAlgorithm(obj, gamma, nRecords, nData, nMapped, obj.hmm1.NStates(), obj.hmm1.NEDists(), obj.epsilon, obj.maxSteps, p, args...)
}

func (obj *HmmEstimator) EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, len(x)); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *HmmEstimator) GetEstimate() (VectorPdf, error) {
	return obj.hmm1, nil
}
