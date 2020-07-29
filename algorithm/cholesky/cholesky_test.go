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

package cholesky

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "testing"
import   "time"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestCholesky1(test *testing.T) {
  n := 4
  a := NewDenseFloat64Matrix([]float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 }, n, n)
  x, _, _ := Run(a)
  r := NewDenseFloat64Matrix([]float64{
     4.24264, 0.00000, 0.00000, 0.00000,
     5.18545, 6.56591, 0.00000, 0.00000,
    12.72792, 3.04604, 1.64974, 0.00000,
    9.89949, 1.62455, 1.84971, 1.39262 }, n, n)
  t := NewFloat64(0.0)

  if t.Mnorm(r.MsubM(x, r)).GetFloat64() > 1e-8 {
    test.Error("Cholesky failed!")
  }
}

func TestCholesky2(test *testing.T) {
  n := 4
  a := NewDenseFloat64Matrix([]float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 }, n, n)
  l, d, _ := Run(a, LDL{true})
  r := NullDenseFloat64Matrix(n, n)
  r.MdotM(r.MdotM(l, d), l.T())
  t := NewFloat64(0.0)

  if t.Mnorm(r.MsubM(r, a)).GetFloat64() > 1e-8 {
    test.Error("Cholesky failed!")
  }
}

// Example 4.7 in
// Gill, Philip E., Walter Murray, and Margaret H. Wright.
// "Practical optimization." (1981).
func TestCholesky3(test *testing.T) {
  n := 3
  a := NewDenseFloat64Matrix([]float64{
    1, 1,       2,
    1, 1+1e-20, 3,
    2, 3,       1 }, n, n)
  l, d, err := Run(a, LDL{true}, ForcePD{true})
  t := NewFloat64(0.0)

  rl := NewDenseFloat64Matrix([]float64{
    1.000000e+00, 0.000000e+00, 0.000000e+00,
    2.651650e-01, 1.000000e+00, 0.000000e+00,
    5.303301e-01, 4.294745e-01, 1.000000e+00 }, n, n)
  rd := NewDenseFloat64Matrix([]float64{
    3.771236e+00, 0.000000e+00, 0.000000e+00,
    0.000000e+00, 5.750446e+00, 0.000000e+00,
    0.000000e+00, 0.000000e+00, 1.121320e+00 }, n, n)
  if err != nil {
    test.Error(err)
  } else {
    if t.Mnorm(rl.MsubM(rl, l)).GetFloat64() > 1e-8 {
      test.Error("Cholesky failed!")
    }
    if t.Mnorm(rd.MsubM(rd, d)).GetFloat64() > 1e-8 {
      test.Error("Cholesky failed!")
    }
  }
}


func TestPerformance(test *testing.T) {
  n := 100

  a1 := NullDenseFloat64Matrix(n, n)
  a1.SetIdentity()
  l1 := NullDenseFloat64Matrix(n, n)
  s1 := NewFloat64(0.0)
  t1 := NewFloat64(0.0)
  inSitu1 := InSitu{l1, nil, s1, t1}

  a2 := NullDenseReal64Matrix(n, n)
  a2.SetIdentity()
  l2 := NullDenseReal64Matrix(n, n)
  s2 := NewReal64(0.0)
  t2 := NewReal64(0.0)
  inSitu2 := InSitu{l2, nil, s2, t2}

  start := time.Now()
  Run(a1, &inSitu1)
  elapsed := time.Since(start)
  fmt.Printf("Cholesky on DenseFloat64Matrix took %s.\n", elapsed)

  start = time.Now()
  Run(a2, &inSitu2)
  elapsed = time.Since(start)
  fmt.Printf("Cholesky on DenseReal64Matrix took %s.\n", elapsed)
}
