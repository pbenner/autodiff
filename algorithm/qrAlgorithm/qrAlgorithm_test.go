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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "sort"
import   "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {
  a := NewMatrix(RealType, 4, 4, []float64{
    1, 2,  3, 4,
    4, 4,  4, 4,
    0, 1, -1, 1,
    0, 0,  2, 3 })

  h, u, _ := Run(a, ComputeU{true})

  b := MdotM(MdotM(u, h), u.T())

  eigenvalues := []float64{-1.561553e+00, -7.416574e-01, 2.561553e+00, 6.741657e+00}

  r := []float64{}
  for i := 0; i < 4; i++ {
    r = append(r, h.At(i, i).GetValue())
  }
  sort.Float64s(r)

  for i := 0; i < 4; i++ {
    if math.Abs(r[i]-eigenvalues[i]) > 1e-5 {
      t.Errorf("test failed for eigenvalue `%d'", i)
    }
  }
  if math.Abs(Mnorm(MsubM(a, b)).GetValue()) > 1e-4 {
    t.Errorf("test failed")
  }
}

func Test2(t *testing.T) {
  a := NewMatrix(RealType, 2, 2, []float64{
    2,  2,
    3, -2 })

  h, _, _ := Run(a)

  r := []float64{}
  for i := 0; i < 2; i++ {
    r = append(r, h.At(i, i).GetValue())
  }
  sort.Float64s(r)

  eigenvalues := []float64{-3.162278e+00, 3.162278e+00}

  for i := 0; i < 2; i++ {
    if math.Abs(r[i]-eigenvalues[i]) > 1e-5 {
      t.Errorf("test failed for eigenvalue `%d'", i)
    }
  }
}

func Test3(t *testing.T) {
  a := NewMatrix(RealType, 16, 16, []float64{
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
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 })

  p, q := splitMatrix(a, 0)

  if p != 4 {
    t.Error("test failed")
  }
  if q != 5 {
    t.Error("test failed")
  }
}

func Test4(t *testing.T) {
  a := NewMatrix(RealType, 4, 4, []float64{
    1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1 })

  p, q := splitMatrix(a, 0)

  if p != 0 {
    t.Error("test failed")
  }
  if q != 4 {
    t.Error("test failed")
  }
}

func Test5(t *testing.T) {
  a := NewMatrix(RealType, 4, 4, []float64{
    1, 1, 1, 1,
    0, 1, 1, 1,
    0, 1, 1, 1,
    0, 0, 1, 1 })

  p, q := splitMatrix(a, 0)

  if p != 1 {
    t.Error("test failed")
  }
  if q != 0 {
    t.Error("test failed")
  }
}

func Test6(t *testing.T) {
  a := NewMatrix(RealType, 5, 5, []float64{
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 1, 1, 1,
    0, 0, 1, 1, 1,
    0, 0, 0, 1, 1 })

  p, q := splitMatrix(a, 0)

  if p != 2 {
    t.Error("test failed")
  }
  if q != 0 {
    t.Error("test failed")
  }
}

func Test7(t *testing.T) {
  a := NewMatrix(RealType, 6, 6, []float64{
     7,  3,  4, -11, -9, -2,
    -6,  4, -5,   7,  1, 12,
    -1, -9,  2,   2,  9,  1,
    -8,  0, -1,   5,  0,  8,
    -4,  3, -5,   7,  2, 10,
     6,  1,  4, -11, -7, -1})

  h, u, _ := Run(a, ComputeU{true})

  b := MdotM(MdotM(u, h), u.T())

  eigenvalues := []float64{5, 5, 4, 3, 1, 1}

  r, _ := Eigenvalues(a)

  for i := 0; i < 4; i++ {
    if math.Abs(r.At(i).GetValue()-eigenvalues[i]) > 1e-5 {
      t.Errorf("test failed for eigenvalue `%d'", i)
    }
  }
  if math.Abs(Mnorm(MsubM(a, b)).GetValue()) > 1e-4 {
    t.Errorf("test failed")
  }
}
