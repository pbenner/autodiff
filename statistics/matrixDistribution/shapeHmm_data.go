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

package matrixDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ShapeHmmDataRecord struct {
	Edist []MatrixPdf
	X     ConstMatrix
}

func (obj ShapeHmmDataRecord) MapIndex(k int) int {
	return k
}

func (obj ShapeHmmDataRecord) GetN() int {
	n, _ := obj.X.Dims()
	return n
}

func (obj ShapeHmmDataRecord) LogPdf(r Scalar, c, k int) error {
	n, m := obj.Edist[c].Dims()

	i := k - n/2
	j := k - n/2 + n

	if l, _ := obj.X.Dims(); i >= 0 && j < l {
		return obj.Edist[c].LogPdf(r, obj.X.ConstSlice(i, j, 0, m))
	} else {
		r.SetValue(0.0)
		return nil
	}
}
