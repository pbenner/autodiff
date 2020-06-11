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

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func (obj *Hmm) bareRealForward(data HmmDataRecord, alpha *DenseBareRealMatrix, t1, t2 *BareReal, n, m int) (*DenseBareRealMatrix, error) {
	// initialize first position
	if n > 0 {
		for i := 0; i < m; i++ {
			if err := data.LogPdf(t2, obj.StateMap[i], 0); err != nil {
				return nil, err
			}
			alpha.At(i, 0).Add(obj.Pi.At(i), t2)
		}
	}
	// loop over x(0), ..., x(N-2)
	for k := 1; k < n-1; k++ {
		// transition to state j
		for j := 0; j < m; j++ {
			at := alpha.AT(j, k)
			// initialize alpha
			at.SetValue(math.Inf(-1))
			// compute:
			// alpha_t(x_j) = sum_i p(x_j | x_i) alpha_s(x_i)
			// transitions from state i
			for i := 0; i < m; i++ {
				t1.Add(obj.Tr.At(i, j), alpha.At(i, k-1))
				at.LOGADD(at, t1, t2)
			}
			// alpha_t(x_j) = p(y_k | x_j) sum_i p(x_j | x_i) alpha_s(x_i)
			if err := data.LogPdf(t2, obj.StateMap[j], k); err != nil {
				return nil, err
			}
			at.ADD(at, t2)
		}
	}
	if n > 1 {
		// last step from x(N-2) to x(N-1)
		// transition to state j
		for j := 0; j < m; j++ {
			at := alpha.AT(j, n-1)
			// initialize alpha
			at.SetValue(math.Inf(-1))
			// compute:
			// alpha_t(x_j) = sum_i p(x_j | x_i) alpha_s(x_i)
			// transitions from state i
			if n > 1 {
				for i := 0; i < m; i++ {
					t1.Add(obj.Tf.At(i, j), alpha.At(i, n-2))
					at.LOGADD(at, t1, t2)
				}
			}
			// alpha_t(x_j) = p(y_k | x_j) sum_i p(x_j | x_i) alpha_s(x_i)
			if err := data.LogPdf(t2, obj.StateMap[j], n-1); err != nil {
				return nil, err
			}
			at.ADD(at, t2)
		}
	}
	return alpha, nil
}

func (obj *Hmm) bareRealBackward(data HmmDataRecord, beta *DenseBareRealMatrix, t1, t2 *BareReal, n, m int) (*DenseBareRealMatrix, error) {
	// initialize last position
	if n > 0 {
		for i := 0; i < m; i++ {
			beta.AT(i, n-1).SetValue(0.0)
		}
	}
	if n > 1 {
		// first step from x(N-1) to x(N-2)
		// transitions from state i
		for i := 0; i < m; i++ {
			bs := beta.AT(i, n-2)
			// initialize beta
			bs.SetValue(math.Inf(-1))
			// compute:
			// beta_s(x_i) = sum_j p(y_{k+1} | x_j) p(x_j | x_i) beta_t(x_j)
			// transition to state j
			for j := 0; j < m; j++ {
				if err := data.LogPdf(t2, obj.StateMap[j], n-1); err != nil {
					return nil, err
				}
				t1.Add(obj.Tf.At(i, j), t2)
				bs.LOGADD(bs, t1, t2)
			}
		}
	}
	// loop over x(N-2), ..., x(0)
	for k := n - 3; k >= 0; k-- {
		// transitions from state i
		for i := 0; i < m; i++ {
			bs := beta.AT(i, k)
			// initialize beta
			bs.SetValue(math.Inf(-1))
			// compute:
			// beta_s(x_i) = sum_j p(y_{k+1} | x_j) p(x_j | x_i) beta_t(x_j)
			// transition to state j
			for j := 0; j < m; j++ {
				t1.Add(obj.Tr.At(i, j), beta.At(j, k+1))
				if err := data.LogPdf(t2, obj.StateMap[j], k+1); err != nil {
					return nil, err
				}
				t1.Add(t1, t2)
				bs.LOGADD(bs, t1, t2)
			}
		}
	}
	return beta, nil
}

func (obj *Hmm) bareRealForwardBackward(data HmmDataRecord, alpha, beta *DenseBareRealMatrix, t1, t2 *BareReal) (*DenseBareRealMatrix, *DenseBareRealMatrix, error) {
	var err error
	// length of the sequence
	n := data.GetN()
	// number of states
	m := obj.M
	// execute forward and backward algorithms
	alpha, err = obj.bareRealForward(data, alpha, t1, t2, n, m)
	if err != nil {
		return nil, nil, err
	}
	beta, err = obj.bareRealBackward(data, beta, t1, t2, n, m)
	if err != nil {
		return nil, nil, err
	}
	return alpha, beta, nil
}
