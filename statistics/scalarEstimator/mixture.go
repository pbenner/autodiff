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

package scalarEstimator

/* -------------------------------------------------------------------------- */

import "fmt"
import "os"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/generic"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type MixtureEstimator struct {
	mixture1   *scalarDistribution.Mixture
	mixture2   *scalarDistribution.Mixture
	mixture3   *scalarDistribution.Mixture
	data       MixtureDataSet
	estimators []ScalarEstimator
	// EM arguments
	epsilon  float64
	maxSteps int
	args     []interface{}
	// hook options
	SaveFile     string
	SaveInterval int
	Trace        string
	Verbose      int
	// estimator options
	OptimizeEmissions bool
	OptimizeWeights   bool
}

func NewMixtureEstimator(weights []float64, estimators []ScalarEstimator, epsilon float64, maxSteps int, args ...interface{}) (*MixtureEstimator, error) {
	if weights == nil {
		weights = make([]float64, len(estimators))
		for i := 0; i < len(estimators); i++ {
			weights[i] = 1.0
		}
	}
	m, err := scalarDistribution.NewMixture(NewVector(BareRealType, weights), nil)
	if err != nil {
		return nil, err
	}
	if len(estimators) != m.NComponents() {
		return nil, fmt.Errorf("invalid number of estimators")
	}
	for i, estimator := range estimators {
		// initialize distribution
		if m.Edist[i] == nil {
			if d, err := estimator.GetEstimate(); err != nil {
				return nil, err
			} else {
				m.Edist[i] = d
			}
		}
	}
	// initialize estimators with data
	r := MixtureEstimator{}
	r.mixture1 = m.Clone()
	r.mixture2 = m.Clone()
	r.mixture3 = m.Clone()
	r.estimators = estimators
	r.epsilon = epsilon
	r.maxSteps = maxSteps
	r.OptimizeEmissions = true
	r.OptimizeWeights = true
	r.args = args
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *MixtureEstimator) GetBasicMixture() generic.BasicMixture {
	return obj.mixture1
}

func (obj *MixtureEstimator) EvaluateLogPdf(pool ThreadPool) error {
	return obj.data.EvaluateLogPdf(obj.mixture2.Edist, pool)
}

func (obj *MixtureEstimator) Swap() {
	obj.mixture1, obj.mixture2, obj.mixture3 = obj.mixture3, obj.mixture1, obj.mixture2
}

func (obj *MixtureEstimator) Emissions(gamma []DenseBareRealVector, p ThreadPool) error {
	mixture1 := obj.mixture1
	mixture2 := obj.mixture2
	// estimate emission parameters
	g := p.NewJobGroup()
	if err := p.AddRangeJob(0, mixture1.NComponents(), g, func(c int, p ThreadPool, erf func() error) error {
		// copy parameters for faster convergence
		p1 := mixture1.Edist[c].GetParameters()
		p2 := mixture2.Edist[c].GetParameters()
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
		if err := mixture1.Edist[c].SetParameters(obj.estimators[c].GetParameters()); err != nil {
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

func (obj *MixtureEstimator) Step(gamma ConstVector, tmp []generic.EmTmp, p ThreadPool) (float64, error) {
	mixture1 := obj.mixture1
	mixture2 := obj.mixture2
	return mixture1.Mixture.EmStep(&mixture1.Mixture, &mixture2.Mixture, obj.data, gamma, tmp, p)
}

/* -------------------------------------------------------------------------- */

func (obj *MixtureEstimator) Clone() *MixtureEstimator {
	estimators := make([]ScalarEstimator, len(obj.estimators))
	for i := 0; i < len(obj.estimators); i++ {
		estimators[i] = obj.estimators[i].CloneScalarEstimator()
	}
	r := MixtureEstimator{}
	r = *obj
	r.mixture1 = r.mixture1.Clone()
	r.mixture2 = r.mixture2.Clone()
	r.mixture3 = r.mixture3.Clone()
	r.estimators = estimators
	return &r
}

func (obj *MixtureEstimator) CloneScalarEstimator() ScalarEstimator {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *MixtureEstimator) ScalarType() ScalarType {
	return obj.mixture1.ScalarType()
}

func (obj *MixtureEstimator) GetParameters() Vector {
	return obj.mixture1.GetParameters()
}

func (obj *MixtureEstimator) SetParameters(parameters Vector) error {
	return obj.mixture1.SetParameters(parameters)
}

func (obj *MixtureEstimator) SetData(x ConstVector, n int) error {
	if data, err := NewMixtureStdDataSet(obj.ScalarType(), x, obj.mixture1.NComponents()); err != nil {
		return err
	} else {
		obj.data = data
	}
	for i, estimator := range obj.estimators {
		// set data
		if err := estimator.SetData(obj.data.GetData(), n); err != nil {
			return err
		}
		// initialize distribution
		if d, err := estimator.GetEstimate(); err != nil {
			return err
		} else {
			obj.mixture1.Edist[i] = d.CloneScalarPdf()
			obj.mixture2.Edist[i] = d.CloneScalarPdf()
			obj.mixture3.Edist[i] = d.CloneScalarPdf()
		}
	}
	return nil
}

func (obj *MixtureEstimator) Estimate(gamma ConstVector, p ThreadPool) error {
	hook_save := generic.EmHook{}
	hook_trace := generic.EmHook{}
	hook_verbose := generic.EmHook{}
	trace := NullVector(obj.ScalarType(), 0)
	if obj.SaveFile != "" && obj.SaveInterval > 0 {
		hook_save.Value = func(mixture generic.BasicMixture, i int, likelihood, epsilon float64) {
			if i%obj.SaveInterval == 0 {
				if d, err := obj.GetEstimate(); err == nil {
					ExportDistribution(obj.SaveFile, d)
				}
			}
		}
	}
	if obj.data.GetCounts() != nil && gamma != nil {
		return fmt.Errorf("cannot nest mixture estimator if data is summarized")
	}
	// add hooks
	//////////////////////////////////////////////////////////////////////////////
	if obj.Trace != "" {
		hook_trace.Value = func(mixture generic.BasicMixture, i int, likelihood, epsilon float64) {
			trace = trace.AppendVector(mixture.GetParameters())
		}
	}
	if obj.Verbose > 1 {
		hook_verbose = generic.DefaultEmHook(os.Stderr)
	} else if obj.Verbose > 0 {
		hook_verbose = generic.PlainEmHook(os.Stderr)
	}
	//////////////////////////////////////////////////////////////////////////////
	args := obj.args
	args = append(args, hook_save)
	args = append(args, hook_trace)
	args = append(args, hook_verbose)
	args = append(args, generic.EmOptimizeEmissions{obj.OptimizeEmissions})
	args = append(args, generic.EmOptimizeWeights{obj.OptimizeWeights})
	//////////////////////////////////////////////////////////////////////////////
	return generic.EmAlgorithm(obj, gamma, obj.data.GetN(), obj.mixture1.NComponents(), obj.epsilon, obj.maxSteps, p, args...)
}

func (obj *MixtureEstimator) EstimateOnData(x, gamma ConstVector, p ThreadPool) error {
	if err := obj.SetData(x, x.Dim()); err != nil {
		return err
	}
	return obj.Estimate(gamma, p)
}

func (obj *MixtureEstimator) GetEstimate() (ScalarPdf, error) {
	return obj.mixture1, nil
}
