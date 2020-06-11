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

package special

/* -------------------------------------------------------------------------- */

import "math"

import . "github.com/pbenner/autodiff/logarithmetic"

/* -------------------------------------------------------------------------- */

type Series interface {
	Eval() float64
}

/* -------------------------------------------------------------------------- */

func SumSeries(series Series, init_value, factor float64, max_terms int) float64 {
	result := 0.0
	for i := 0; i < max_terms; i++ {
		next_term := series.Eval()
		result += next_term
		if math.Abs(factor*result) >= math.Abs(next_term) {
			break
		}
	}
	return result
}

func SumLogSeries(series Series, init_value, logFactor float64, max_terms int) float64 {
	result := math.Inf(-1)
	for i := 0; i < max_terms; i++ {
		next_term := series.Eval()
		result = LogAdd(result, next_term)
		if logFactor+result >= next_term {
			break
		}
	}
	return result
}
