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
import "io"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type BaumWelchHook struct {
	Value func(hmm BasicHmm, i int, likelihood, epsilon float64)
}

type BaumWelchOptimizeEmissions struct {
	Value bool
}

type BaumWelchOptimizeTransitions struct {
	Value bool
}

type BaumWelchTmp struct {
	alpha      *DenseBareRealMatrix
	beta       *DenseBareRealMatrix
	xi         *DenseBareRealMatrix
	xiz        *BareReal
	gamma      []DenseBareRealVector
	gamma0     DenseBareRealVector
	gammaTmp   DenseBareRealVector
	t1         *BareReal
	t2         *BareReal
	t3         *BareReal
	tr         *DenseBareRealMatrix
	pi         DenseBareRealVector
	likelihood float64
	init       bool
}

func DefaultBaumWelchHook(writer io.Writer) BaumWelchHook {
	hook := BaumWelchHook{}
	hook.Value = func(hmm BasicHmm, i int, likelihood, epsilon float64) {
		if i == 0 || i == 1 {
			fmt.Fprintf(writer, "Baum-Welch iteration %v\n", i)
		} else {
			fmt.Fprintf(writer, "Baum-Welch iteration %v (epsilon=%f)\n", i, epsilon)
		}
		fmt.Fprintf(writer, "----------------------------------------\n")
		if !math.IsNaN(likelihood) {
			fmt.Fprintf(writer, "log Pdf(x) = %v\n\n", likelihood)
		}
		fmt.Fprintf(writer, "%v\n", hmm)
		fmt.Fprintf(writer, "\n")
	}
	return hook
}

func PlainBaumWelchHook(writer io.Writer) BaumWelchHook {
	hook := BaumWelchHook{}
	hook.Value = func(hmm BasicHmm, i int, likelihood, epsilon float64) {
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

type baumWelchCore interface {
	EvaluateLogPdf(pool ThreadPool) error
	GetBasicHmm() BasicHmm
	Swap()
	Step(meta ConstVector, tmp []BaumWelchTmp, p ThreadPool) (float64, error)
	Emissions(gamma []DenseBareRealVector, p ThreadPool) error
}

/* -------------------------------------------------------------------------- */

func baumWelchAlgorithm(obj baumWelchCore, meta ConstVector, tmp []BaumWelchTmp, epsilon float64, maxSteps int, hooks []BaumWelchHook, p ThreadPool) error {
	for _, hook := range hooks {
		if hook.Value != nil {
			hook.Value(obj.GetBasicHmm(), 0, math.NaN(), math.NaN())
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
		// update hmm1
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
					hook.Value(obj.GetBasicHmm(), k+1, likelihood_new, likelihood_new-likelihood_old)
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

func BaumWelchAlgorithm(obj baumWelchCore, meta ConstVector, nRecords, nData, nMapped, nStates, nEdists int, epsilon float64, maxSteps int, p ThreadPool, args ...interface{}) error {
	if nRecords == 0 {
		return nil
	}
	// declare optional arguments
	hooks := []BaumWelchHook{}
	optimizeEmissions := true
	optimizeTransitions := true
	// parse optional arguments
	for _, arg := range args {
		switch a := arg.(type) {
		case BaumWelchHook:
			hooks = append(hooks, a)
		case BaumWelchOptimizeEmissions:
			optimizeEmissions = a.Value
		case BaumWelchOptimizeTransitions:
			optimizeTransitions = a.Value
		}
	}
	threads := p.NumberOfThreads()
	// number of states
	m1 := nStates
	m2 := nEdists
	// allocate memory
	tmp := make([]BaumWelchTmp, threads)
	for threadIdx := 0; threadIdx < threads; threadIdx++ {
		// forward and backward probabilities
		tmp[threadIdx].alpha = NullDenseBareRealMatrix(m1, nData)
		tmp[threadIdx].beta = NullDenseBareRealMatrix(m1, nData)
		// initial probabilities
		tmp[threadIdx].pi = NullDenseBareRealVector(m1)
		// transition matrix
		if optimizeTransitions {
			tmp[threadIdx].tr = NullDenseBareRealMatrix(m1, m1)
			tmp[threadIdx].xi = NullDenseBareRealMatrix(m1, m1)
			tmp[threadIdx].xiz = NullBareReal()
		}
		if optimizeEmissions {
			tmp[threadIdx].gamma = make([]DenseBareRealVector, m2)
			for c := 0; c < m2; c++ {
				tmp[threadIdx].gamma[c] = NullDenseBareRealVector(nMapped)
			}
			tmp[threadIdx].gammaTmp = NullDenseBareRealVector(m1)
		}
		tmp[threadIdx].gamma0 = NullDenseBareRealVector(m1)
		// some temporary variables
		tmp[threadIdx].t1 = NewBareReal(0.0)
		tmp[threadIdx].t2 = NewBareReal(0.0)
		tmp[threadIdx].t3 = NewBareReal(0.0)
	}
	return baumWelchAlgorithm(obj, meta, tmp, epsilon, maxSteps, hooks, p)
}
