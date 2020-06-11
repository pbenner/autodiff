/* Copyright (C) 2015 Philipp Benner
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

package blahut

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func flatten(m [][]float64) []float64 {
	v := []float64{}
	for i, _ := range m {
		v = append(v, m[i]...)
	}
	return v
}

func normalizeSlice(p []float64) {
	sum := 0.0
	for _, v := range p {
		sum += v
	}
	for i, _ := range p {
		p[i] /= sum
	}
}

func normalizeVector(p Vector) {
	sum := NewScalar(p.ElementType(), 0.0)
	for i := 0; i < p.Dim(); i++ {
		sum.Add(sum, p.At(i))
	}
	for i := 0; i < p.Dim(); i++ {
		p.At(i).Div(p.At(i), sum)
	}
}
