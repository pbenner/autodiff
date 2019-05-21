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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/logarithmetic"

/* -------------------------------------------------------------------------- */

type LogisticRegressionBare struct {
  Theta []float64
}

/* -------------------------------------------------------------------------- */

func (obj LogisticRegressionBare) Dim() int {
  return len(obj.Theta)-1
}

func (obj LogisticRegressionBare) LogPdf(x []float64, index []int) float64 {
  // set r to first element of theta
  r := obj.Theta[0]
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  for ; i < n; i++ {
    r += x[i]*obj.Theta[index[i]]
  }
  return -LogAdd(0.0, -r)
}
