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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "sort"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
    1, 2,  3, 4,
    4, 4,  4, 4,
    0, 1, -1, 1,
    0, 0,  2, 3 }, 4, 4)

  h, u, _ := Run(a, ComputeU{true})

  b := NullDenseFloat64Matrix(4, 4)
  b.MdotM(b.MdotM(u, h), u.T())

  if math.Abs(t.Mnorm(a.MsubM(a, b)).GetFloat64()) > 1e-4 {
    test.Errorf("test failed")
  }
}

func Test2(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    2,  2,
    3, -2 }, 2, 2)

  h, u, _ := Run(a)

  r := []float64{}
  for i := 0; i < 2; i++ {
    r = append(r, h.At(i, i).GetFloat64())
  }
  sort.Float64s(r)

  eigenvalues := []float64{-3.162278e+00, 3.162278e+00}

  for i := 0; i < 2; i++ {
    if math.Abs(r[i]-eigenvalues[i]) > 1e-5 {
      test.Errorf("test failed for eigenvalue `%d'", i)
    }
  }
  if u != nil {
    test.Error("test failed")
  }
}

func Test3(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 }, 16, 16)

  p, q := splitMatrix(a, 0)

  if p != 4 {
    test.Error("test failed")
  }
  if q != 5 {
    test.Error("test failed")
  }
}

func Test4(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1 }, 4, 4)

  p, q := splitMatrix(a, 0)

  if p != 0 {
    test.Error("test failed")
  }
  if q != 4 {
    test.Error("test failed")
  }
}

func Test5(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 1, 1,
    0, 1, 1, 1,
    0, 1, 1, 1,
    0, 0, 1, 1 }, 4, 4)

  p, q := splitMatrix(a, 0)

  if p != 1 {
    test.Error("test failed")
  }
  if q != 0 {
    test.Error("test failed")
  }
}

func Test6(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 1, 1, 1,
    0, 0, 1, 1, 1,
    0, 0, 0, 1, 1 }, 5, 5)

  p, q := splitMatrix(a, 0)

  if p != 2 {
    test.Error("test failed")
  }
  if q != 0 {
    test.Error("test failed")
  }
}

func Test7(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
     7,  3,  4, -11, -9, -2,
    -6,  4, -5,   7,  1, 12,
    -1, -9,  2,   2,  9,  1,
    -8,  0, -1,   5,  0,  8,
    -4,  3, -5,   7,  2, 10,
    6,  1,  4, -11, -7, -1}, 6, 6)

  h, u, _ := Run(a, ComputeU{true})

  b := NullDenseFloat64Matrix(6, 6)
  b.MdotM(b.MdotM(u, h), u.T())

  if math.Abs(t.Mnorm(a.MsubM(a, b)).GetFloat64()) > 1e-4 {
    test.Errorf("test failed")
  }
}

func Test8(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
     20.2,   0.0,  10.8, -25.5, -12.8, -19.7,
      0.0,  11.6,  -3.7,  -1.3, -10.5,   6.2,
     10.8,  -3.7,   8.7, -15.9,  -6.1, -12.8,
    -25.5,  -1.3, -15.9,  36.9,  21.5,  22.9,
    -12.8, -10.5,  -6.1,  21.5,  21.6,   6.6,
    -19.7,   6.2, -12.8,  22.9,   6.6,  31.4 }, 6, 6)

  h1, u1, _ := Run(a, ComputeU{true}, Symmetric{true })
  h2, u2, _ := Run(a, ComputeU{true}, Symmetric{false})

  b1 := NullDenseFloat64Matrix(6, 6)
  b1.MdotM(b1.MdotM(u1, h1), u1.T())
  b2 := NullDenseFloat64Matrix(6, 6)
  b2.MdotM(b2.MdotM(u2, h2), u2.T())

  if math.Abs(t.Mnorm(b1.MsubM(a, b1)).GetFloat64()) > 1e-4 {
    test.Errorf("test failed")
  }
  if math.Abs(t.Mnorm(b2.MsubM(a, b2)).GetFloat64()) > 1e-4 {
    test.Errorf("test failed")
  }
  v1 := h1.Diag(); v1.Sort(true)
  v2 := h2.Diag(); v2.Sort(true)
  if math.Abs(t.Vnorm(v1.VsubV(v1, v2)).GetFloat64()) > 1e-4 {
    test.Errorf("test failed")
  }
}
