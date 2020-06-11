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

package autodiff

/* -------------------------------------------------------------------------- */

import "fmt"
import "testing"
import "time"

/* -------------------------------------------------------------------------- */

func TestMatrixPerformance(t *testing.T) {

	n := 1000

	{
		a := NullDenseRealMatrix(n, n)
		a.Map(func(x Scalar) { x.SetValue(2.0) })

		start := time.Now()
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a.At(i, j).Add(a.At(i, j), a.At(0, 0))
				a.At(i, j).Sub(a.At(i, j), a.At(0, 0))
				a.At(i, j).Mul(a.At(i, j), a.At(0, 0))
				a.At(i, j).Div(a.At(i, j), a.At(0, 0))
			}
		}
		elapsed := time.Since(start)
		fmt.Printf("Operations on DenseRealMatrix took %s.\n", elapsed)

		start = time.Now()
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a.AT(i, j).ADD(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).SUB(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).MUL(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).DIV(a.AT(i, j), a.AT(0, 0))
			}
		}
		elapsed = time.Since(start)
		fmt.Printf("Operations on DenseRealMatrix with concrete types took %s.\n", elapsed)
	}

	{
		a := NullDenseBareRealMatrix(n, n)
		a.Map(func(x Scalar) { x.SetValue(2.0) })

		start := time.Now()
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a.At(i, j).Add(a.At(i, j), a.At(0, 0))
				a.At(i, j).Sub(a.At(i, j), a.At(0, 0))
				a.At(i, j).Mul(a.At(i, j), a.At(0, 0))
				a.At(i, j).Div(a.At(i, j), a.At(0, 0))
			}
		}
		elapsed := time.Since(start)
		fmt.Printf("Operations on DenseBareRealMatrix took %s.\n", elapsed)

		start = time.Now()
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a.AT(i, j).ADD(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).SUB(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).MUL(a.AT(i, j), a.AT(0, 0))
				a.AT(i, j).DIV(a.AT(i, j), a.AT(0, 0))
			}
		}
		elapsed = time.Since(start)
		fmt.Printf("Operations on DenseBareRealMatrix with concrete types took %s.\n", elapsed)
	}

}
