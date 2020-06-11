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
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func (obj *Hmm) baumWelchThread(hmm1, hmm2 *Hmm, data HmmDataRecord, meta ConstVector, tmp *BaumWelchTmp, p ThreadPool) error {
	n := data.GetN()
	m := obj.M
	// get temporary memory
	alpha := tmp.alpha
	beta := tmp.beta
	xi := tmp.xi
	xiz := tmp.xiz
	gamma := tmp.gamma
	gamma0 := tmp.gamma0
	gammaTmp := tmp.gammaTmp
	t1 := tmp.t1
	t2 := tmp.t2
	t3 := tmp.t3
	pi := tmp.pi
	tr := tmp.tr
	// reset variables if this is the first time this
	// thread is executed
	if tmp.init == false {
		pi.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
		tr.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
		for c := 0; c < len(gamma); c++ {
			gamma[c].Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
		}
		tmp.likelihood = 0.0
		tmp.init = true
	}
	// execute forward-backward algorithm
	hmm2.bareRealForwardBackward(data, alpha, beta, t1, t2)
	// compute gamma at position 0
	{
		// normalization constant
		t1.SetValue(math.Inf(-1))
		for i := 0; i < m; i++ {
			gamma0.AT(i).ADD(alpha.AT(i, 0), beta.AT(i, 0))
			t1.LOGADD(t1, gamma0.AT(i), t3)
		}
		if math.IsInf(t1.GetValue(), -1) {
			return fmt.Errorf("all paths have zero probability")
		}
		// normalize gamma0
		for i := 0; i < m; i++ {
			gamma0.AT(i).Sub(gamma0.AT(i), t1)
		}
		// update pi
		for i := 0; i < m; i++ {
			pi.AT(i).LOGADD(pi.AT(i), gamma0.AT(i), t3)
		}
	}
	// compute gamma temporaries
	if gamma != nil {
		for k := 0; k < n; k++ {
			// normalization constant
			t1.SetValue(math.Inf(-1))
			for i := 0; i < m; i++ {
				gammaTmp.AT(i).ADD(alpha.AT(i, k), beta.AT(i, k))
				t1.LOGADD(t1, gammaTmp.AT(i), t3)
			}
			if math.IsInf(t1.GetValue(), -1) {
				return fmt.Errorf("all paths have zero probability")
			}
			for i := 0; i < m; i++ {
				c := obj.StateMap[i]
				l := data.MapIndex(k)
				// normalize gamma
				gammaTmp.AT(i).Sub(gammaTmp.AT(i), t1)
				if meta != nil {
					gammaTmp.AT(i).Add(gammaTmp.AT(i), meta.ConstAt(l))
				}
				// sum up gamma
				gamma[c].AT(l).LOGADD(gamma[c].AT(l), gammaTmp.AT(i), t3)
			}
		}
	}
	if xi != nil {
		// compute xi and update transition matrix
		for k := 0; k < n-1; k++ {
			if k == n-2 && obj.finalStates != nil {
				// skip last transition if a final state is set
				break
			}
			// reset normalization constant for xi
			xiz.SetValue(math.Inf(-1))
			// compute xi temporaries and update parameters
			for i := 0; i < m; i++ {
				// compute xi temporaries (to save memory xi is not fully evaluated)
				for j := 0; j < m; j++ {
					c := obj.StateMap[j]
					t := xi.AT(i, j)
					t.Add(alpha.At(i, k), hmm2.Tr.At(i, j))
					t.ADD(t, beta.AT(j, k+1))
					if err := data.LogPdf(t3, c, k+1); err != nil {
						return err
					}
					t.ADD(t, t3)
					// sum up values for normalization
					xiz.LOGADD(xiz, t, t3)
				}
			}
			// update transition matrix
			for i := 0; i < m; i++ {
				for j := 0; j < m; j++ {
					t := tr.AT(i, j)
					s := xi.AT(i, j)
					// normalize xi
					s.SUB(s, xiz)
					// sum up xi
					t.LOGADD(t, xi.AT(i, j), t3)
				}
			}
		}
	}
	// compute log-likelihood
	t1.SetValue(math.Inf(-1))
	for i := 0; i < m; i++ {
		t1.LOGADD(t1, alpha.AT(i, n-1), t3)
	}
	tmp.likelihood += t1.GetValue()
	return nil
}

func (obj *Hmm) BaumWelchStep(hmm1, hmm2 *Hmm, data HmmDataSet, meta ConstVector, tmp []BaumWelchTmp, p ThreadPool) (float64, error) {
	if obj.finalStates != nil && len(obj.finalStates) > 1 {
		return math.Inf(-1), fmt.Errorf("cannot optimize models with more than one final state")
	}
	m := obj.M
	// tell every thread that it needs to reset all variables
	for threadIdx := 0; threadIdx < len(tmp); threadIdx++ {
		tmp[threadIdx].init = false
	}
	g := p.NewJobGroup()
	// loop over sequences
	for d_ := 0; d_ < data.GetNRecords(); d_++ {
		// make a thread-safe copy of d
		d := d_
		if err := p.AddJob(g, func(p ThreadPool, erf func() error) error {
			if erf() != nil {
				return nil
			}
			r := data.GetRecord(d)
			return obj.baumWelchThread(hmm1, hmm2, r, meta, &tmp[p.GetThreadId()], p)
		}); err != nil {
			return math.Inf(-1), err
		}
	}
	// set pi and the transition matrix to zero
	hmm1.Pi.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
	if tmp[0].tr != nil {
		hmm1.Tr.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
	}
	// wait for all threads to finish
	if err := p.Wait(g); err != nil {
		return math.Inf(-1), nil
	}
	// get some temporary variables
	t1 := tmp[0].t1
	t2 := tmp[0].t2
	t3 := tmp[0].t3
	// merge contributions from all threads
	for threadIdx := 0; threadIdx < len(tmp); threadIdx++ {
		if tmp[threadIdx].init == false {
			// this thread was never used
			continue
		}
		for i := 0; i < m; i++ {
			hmm1.Pi.At(i).LogAdd(hmm1.Pi.At(i), tmp[threadIdx].pi.At(i), t3)
		}
	}
	if tmp[0].tr != nil {
		for threadIdx := 0; threadIdx < len(tmp); threadIdx++ {
			if tmp[threadIdx].init == false {
				// this thread was never used
				continue
			}
			for i := 0; i < m; i++ {
				for j := 0; j < m; j++ {
					t := hmm1.Tr.At(i, j)
					t.LogAdd(t, tmp[threadIdx].tr.At(i, j), t3)
				}
			}
		}
	}
	if tmp[0].init == false {
		for c := 0; c < len(tmp[0].gamma); c++ {
			tmp[0].gamma[c].Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
		}
		tmp[0].likelihood = 0.0
		tmp[0].init = true
	}
	// merge gamma variables and log-likelihoods
	for threadIdx := 1; threadIdx < len(tmp); threadIdx++ {
		if tmp[threadIdx].init == false {
			// this thread was never used
			continue
		}
		for c := 0; c < len(tmp[0].gamma); c++ {
			for l := 0; l < data.GetNMapped(); l++ {
				tmp[0].gamma[c].AT(l).LOGADD(
					tmp[0].gamma[c].AT(l),
					tmp[threadIdx].gamma[c].AT(l), t3)
			}
		}
		tmp[0].likelihood += tmp[threadIdx].likelihood
	}
	// normalize pi and the transition matrix
	if err := hmm1.normalize(t1, t2); err != nil {
		return math.Inf(-1), err
	}
	return tmp[0].likelihood, nil
}
