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

package cholesky

/* -------------------------------------------------------------------------- */

//import   "fmt"

import   "testing"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestCholesky1(t *testing.T) {
  n := 4
  a := NewMatrix(RealType, n, n, []float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 })
  x, _, _ := Run(a)
  r := NewMatrix(RealType, n, n, []float64{
     4.24264, 0.00000, 0.00000, 0.00000,
     5.18545, 6.56591, 0.00000, 0.00000,
    12.72792, 3.04604, 1.64974, 0.00000,
     9.89949, 1.62455, 1.84971, 1.39262 })

  if Mnorm(MsubM(x, r)).GetValue() > 1e-8 {
    t.Error("Cholesky failed!")
  }
}

func TestCholesky2(t *testing.T) {
  n := 4
  a := NewMatrix(RealType, n, n, []float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 })
  s := NewInSitu(RealType, n, true)
  l, d, _ := Run(a, &s, LDL{true})
  r := MdotM(MdotM(l, d), l.T())

  if Mnorm(MsubM(r, a)).GetValue() > 1e-8 {
    t.Error("Cholesky failed!")
  }
}

// Example 4.7 in
// Gill, Philip E., Walter Murray, and Margaret H. Wright.
// "Practical optimization." (1981).
func TestCholesky3(t *testing.T) {
  n := 3
  a := NewMatrix(RealType, n, n, []float64{
    1, 1,       2,
    1, 1+1e-20, 3,
    2, 3,       1 })
  s := NewInSitu(RealType, n, true)
  l, d, err := Run(a, &s, LDL{true}, ForcePD{true})

  rl := NewMatrix(RealType, n, n, []float64{
    1.000000e+00, 0.000000e+00, 0.000000e+00,
    2.651650e-01, 1.000000e+00, 0.000000e+00,
    5.303301e-01, 4.294745e-01, 1.000000e+00 })
  rd := NewMatrix(RealType, n, n, []float64{
    3.771236e+00, 0.000000e+00, 0.000000e+00,
    0.000000e+00, 5.750446e+00, 0.000000e+00,
    0.000000e+00, 0.000000e+00, 1.121320e+00 })
  if err != nil {
    t.Error(err)
  } else {
    if Mnorm(MsubM(rl, l)).GetValue() > 1e-8 {
      t.Error("Cholesky failed!")
    }
    if Mnorm(MsubM(rd, d)).GetValue() > 1e-8 {
      t.Error("Cholesky failed!")
    }
  }
}
