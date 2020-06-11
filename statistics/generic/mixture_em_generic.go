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

package generic

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"
import "io"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type EmTmp struct {
	gammaTmp   DenseBareRealVector
	gamma      []DenseBareRealVector
	logWeights DenseBareRealVector
	t1         *BareReal
	t2         *BareReal
	likelihood float64
	init       bool
}

/* -------------------------------------------------------------------------- */

type EmHook struct {
	Value func(mixture BasicMixture, i int, likelihood, epsilon float64)
}

type EmOptimizeEmissions struct {
	Value bool
}

type EmOptimizeWeights struct {
	Value bool
}

func DefaultEmHook(writer io.Writer) EmHook {
	hook := EmHook{}
	hook.Value = func(mixture BasicMixture, i int, likelihood, epsilon float64) {
		if i == 0 || i == 1 {
			fmt.Fprintf(writer, "EM iteration %v\n", i)
		} else {
			fmt.Fprintf(writer, "EM iteration %v (epsilon=%f)\n", i, epsilon)
		}
		fmt.Fprintf(writer, "----------------------------------------\n")
		if !math.IsNaN(likelihood) {
			fmt.Fprintf(writer, "log Pdf(x) = %v\n\n", likelihood)
		}
		fmt.Fprintf(writer, "%v\n", mixture)
		fmt.Fprintf(writer, "\n")
	}
	return hook
}

func PlainEmHook(writer io.Writer) EmHook {
	hook := EmHook{}
	hook.Value = func(mixture BasicMixture, i int, likelihood, epsilon float64) {
		switch i {
		case 0:
			fmt.Fprintf(writer, "%10s %20s %18s\n", "Iteration", "Log Likelihood", "Change")
		case 1:
			fmt.Fprintf(writer, "%10d %20f %18s\n", i, likelihood, "-")
		default:
			fmt.Fprintf(writer, "%10d %20f %18f\n", i, likelihood, epsilon)
		}
	}
	return hook
}

/* -------------------------------------------------------------------------- */

type emCore interface {
	EvaluateLogPdf(p ThreadPool) error
	GetBasicMixture() BasicMixture
	Swap()
	Step(meta ConstVector, tmp []EmTmp, p ThreadPool) (float64, error)
	Emissions(gamma []DenseBareRealVector, p ThreadPool) error
}

/* -------------------------------------------------------------------------- */

func emAlgorithm(obj emCore, meta ConstVector, tmp []EmTmp, epsilon float64, maxSteps int, hooks []EmHook, p ThreadPool) error {
	for _, hook := range hooks {
		if hook.Value != nil {
			hook.Value(obj.GetBasicMixture(), 0, math.NaN(), math.NaN())
		}
	}
	// perform a single step if this is a nested Em
	if meta != nil {
		maxSteps = 1
	}
	likelihood_old := math.Inf(-1)

	for k := 0; maxSteps == -1 || k < maxSteps; k++ {
		// swap both distributions
		obj.Swap()
		// initialize px
		if err := obj.EvaluateLogPdf(p); err != nil {
			return err
		}
		// update mixture1
		if likelihood_new, err := obj.Step(meta, tmp, p); err != nil {
			return err
		} else {
			if tmp[0].gamma != nil {
				if err := obj.Emissions(tmp[0].gamma, p); err != nil {
					return err
				}
			}
			for _, hook := range hooks {
				if hook.Value != nil {
					hook.Value(obj.GetBasicMixture(), k+1, likelihood_new, likelihood_new-likelihood_old)
				}
			}
			// check convergence (and cycles)
			if likelihood_new-likelihood_old < epsilon {
				break
			}
			likelihood_old = likelihood_new
		}
	}
	return nil
}

func EmAlgorithm(obj emCore, meta ConstVector, nData, nComponents int, epsilon float64, maxSteps int, p ThreadPool, args ...interface{}) error {
	// gamma values used for hierarchical EM algorithms
	hooks := []EmHook{}
	optimizeEmissions := true
	optimizeWeights := true
	// parse optional arguments
	for _, arg := range args {
		switch a := arg.(type) {
		case EmHook:
			hooks = append(hooks, a)
		case EmOptimizeEmissions:
			optimizeEmissions = a.Value
		case EmOptimizeWeights:
			optimizeWeights = a.Value
		}
	}
	threads := p.NumberOfThreads()
	n := nData
	m := nComponents
	// allocate memory
	tmp := make([]EmTmp, threads)
	for threadIdx := 0; threadIdx < threads; threadIdx++ {
		tmp[threadIdx].gammaTmp = NullDenseBareRealVector(m)
		if optimizeWeights {
			tmp[threadIdx].logWeights = NullDenseBareRealVector(m)
		}
		// some temporary variables
		tmp[threadIdx].t1 = NewBareReal(0.0)
		tmp[threadIdx].t2 = NewBareReal(0.0)
		if optimizeEmissions {
			tmp[threadIdx].gamma = make([]DenseBareRealVector, m)
			for i := 0; i < m; i++ {
				tmp[threadIdx].gamma[i] = NullDenseBareRealVector(n)
			}
		}
	}
	return emAlgorithm(obj, meta, tmp, epsilon, maxSteps, hooks, p)
}
