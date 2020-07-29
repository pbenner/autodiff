/* Copyright (C) 2015-2020 Philipp Benner
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

package gradientDescent

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(test *testing.T) {
  m1 := NewDenseReal64Matrix([]float64{1,2,3,4}, 2, 2)
  m2 := m1.CloneMatrix()
  m3 := NewDenseReal64Matrix([]float64{-2, 1, 1.5, -0.5}, 2, 2)
  s  := NewReal64 (0.0)
  t  := NewFloat64(0.0)

  rows, cols := m1.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := NullDenseFloat64Matrix(rows, rows)
  I.SetIdentity()
  // objective function
  f := func(x ConstVector) (MagicScalar, error) {
    m2.AsVector().Set(x)
    s.Mnorm(m2.MsubM(m2.MdotM(m1, m2), I))
    return s, nil
  }
  x, _ := Run(f, m2.AsVector(), 0.01)
  m2.AsVector().Set(x)

  if t.Mnorm(m2.MsubM(m2, m3)).GetFloat64() > 1e-8 {
    test.Error("Inverting matrix failed!")
  }
}
