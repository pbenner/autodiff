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

func (obj *Hmm) Viterbi(data HmmDataRecord) ([]int, error) {
	n := data.GetN()
	if n == 0 {
		return nil, nil
	}
	// number of states
	m := obj.M
	// temporary variables
	t1 := make([][]float64, m)
	t2 := make([][]int, m)
	t3 := NullBareReal()
	for i := 0; i < m; i++ {
		t1[i] = make([]float64, n)
		t2[i] = make([]int, n)
	}
	// result
	r := make([]int, n)
	// initialize tables at x(0)
	for j := 0; j < m; j++ {
		if err := data.LogPdf(t3, obj.StateMap[j], 0); err != nil {
			return nil, err
		}
		t1[j][0] = obj.Pi.At(j).GetValue() + t3.GetValue()
		t2[j][0] = 0
	}
	// loop over x(1), ..., x(N-1)
	for k := 1; k < n-1; k++ {
		for j := 0; j < m; j++ {
			i_pos := 0
			i_val := math.Inf(-1)
			// loop over states and find maximum
			for i := 0; i < m; i++ {
				if v := t1[i][k-1] + obj.Tr.At(i, j).GetValue(); v > i_val {
					i_pos = i
					i_val = v
				}
			}
			if err := data.LogPdf(t3, obj.StateMap[j], k); err != nil {
				return nil, err
			}
			t1[j][k] = i_val + t3.GetValue()
			t2[j][k] = i_pos
		}
	}
	// last transition
	if n > 1 {
		k := n - 1
		for j := 0; j < m; j++ {
			i_pos := 0
			i_val := math.Inf(-1)
			// loop over states and find maximum
			for i := 0; i < m; i++ {
				if v := t1[i][k-1] + obj.Tf.At(i, j).GetValue(); v > i_val {
					i_pos = i
					i_val = v
				}
			}
			if err := data.LogPdf(t3, obj.StateMap[j], k); err != nil {
				return nil, err
			}
			t1[j][k] = i_val + t3.GetValue()
			t2[j][k] = i_pos
		}
	}
	// loop backwards
	// find maximum at the last position
	i_pos := 0
	i_val := math.Inf(-1)
	for i := 0; i < m; i++ {
		// loop over states and find maximum
		if t1[i][n-1] > i_val {
			i_pos = i
			i_val = t1[i][n-1]
		}
	}
	r[n-1] = i_pos
	for k := n - 2; k >= 0; k-- {
		r[k] = t2[r[k+1]][k+1]
	}
	return r, nil
}
