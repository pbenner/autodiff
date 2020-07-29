/* Copyright (C) 2017-2020 Philipp Benner
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

package svd

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "sort"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0,
    0, 2, 1, 0, 0,
    0, 0, 3, 1, 0,
    0, 0, 0, 4, 1,
    0, 0, 0, 0, 5 }, 5, 5)

  h, _, _, _ := Run(a)

  r := []float64{}
  for i := 0; i < 4; i++ {
    r = append(r, h.At(i, i).GetFloat64())
  }
  sort.Float64s(r)

  singularValues := []float64{8.584972e-01, 2.110322e+00, 3.086113e+00, 4.086174e+00, 5.252486e+00}

  for i := 0; i < len(r); i++ {
    if math.Abs(r[i]-singularValues[i]) > 1e-5 {
      test.Errorf("test failed for singular value `%d'", i)
    }
  }
}

func Test2(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0,
    0, 2, 1,
    0, 0, 3 }, 3, 3)

  h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

  d := NullDenseFloat64Matrix(3, 3)
  d.MdotM(d.MdotM(u.T(), a), v)

  if t.Mnorm(d.MsubM(d, h)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}

func Test3(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0,
    0, 2, 1, 0, 0,
    0, 0, 3, 1, 0,
    0, 0, 0, 4, 1,
    0, 0, 0, 0, 5,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0 }, 7, 5)

  h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

  d := NullDenseFloat64Matrix(7, 5)
  b := NullDenseFloat64Matrix(7, 5)
  d.MdotM(d.MdotM(u.T(), a), v)
  b.MdotM(b.MdotM(u, h), v.T())

  if t.Mnorm(d.MsubM(d, h)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
  if t.Mnorm(b.MsubM(a, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}

func Test4(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0, 0, 0,
    0, 2, 1, 0, 0, 0, 0,
    0, 0, 3, 1, 0, 0, 0,
    0, 0, 0, 4, 1, 0, 0,
    0, 0, 0, 0, 1e-21, 1, 0,
    0, 0, 0, 0, 0, 6, 1,
    0, 0, 0, 0, 0, 0, 7 }, 7, 7)

  h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

  d := NullDenseFloat64Matrix(7, 7)
  b := NullDenseFloat64Matrix(7, 7)
  d.MdotM(d.MdotM(u.T(), a), v)
  b.MdotM(b.MdotM(u, h), v.T())

  if t.Mnorm(d.MsubM(d, h)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
  if t.Mnorm(b.MsubM(a, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}

func Test5(test *testing.T) {
  t := NewFloat64(0.0)
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0, 0, 0,
    0, 2, 1, 0, 0, 0, 0,
    0, 0, 3, 1, 0, 0, 0,
    0, 0, 0, 4, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 6, 1,
    0, 0, 0, 0, 0, 0, 7 }, 7, 7)

  h, u, v, _ := Run(a, ComputeU{true}, ComputeV{true})

  d := NullDenseFloat64Matrix(7, 7)
  b := NullDenseFloat64Matrix(7, 7)
  d.MdotM(d.MdotM(u.T(), a), v)
  b.MdotM(b.MdotM(u, h), v.T())

  if t.Mnorm(d.MsubM(d, h)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
  if t.Mnorm(b.MsubM(a, b)).GetFloat64() > 1e-8 {
    test.Error("test failed")
  }
}

func Test6(test *testing.T) {
  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, 16, 16)

  p, q := splitMatrix(a, 0)

  if p != 6 {
    test.Error("test failed")
  }
  if q != 3 {
    test.Error("test failed")
  }
}
