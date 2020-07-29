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

package givensRotation

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {

  a := NewDenseFloat64Matrix([]float64{
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 4, 4, 4, 4, 4, 4, 4, 4,
    0, 0, 0, 5, 5, 5, 5, 5, 5, 5,
    0, 0, 0, 0, 6, 6, 6, 6, 6, 6,
    0, 0, 0, 0, 0, 7, 7, 7, 7, 7,
    0, 0, 0, 0, 0, 0, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 9, 9, 9,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 12, 10)
  _, n := a.Dims()

  c  := NewFloat64(3.0)
  s  := NewFloat64(2.0)
  t1 := NewFloat64(0.0)
  t2 := NewFloat64(0.0)

  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyHessenbergLeft(a1, c, s, i, j, t1, t2)
      ApplyLeft          (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyHessenbergRight(a1, c, s, i, j, t1, t2)
      ApplyRight          (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
}

func Test2(test *testing.T) {

  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 2, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 4, 4, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 5, 5, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 6, 6, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 7, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 8, 8, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 9, 9,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 12, 10)
  _, n := a.Dims()

  c  := NewFloat64(3.0)
  s  := NewFloat64(2.0)
  t1 := NewFloat64(0.0)
  t2 := NewFloat64(0.0)

  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyBidiagLeft(a1, c, s, i, j, t1, t2)
      ApplyLeft      (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyBidiagRight(a1, c, s, i, j, t1, t2)
      ApplyRight      (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
}

func Test3(test *testing.T) {

  a := NewDenseFloat64Matrix([]float64{
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 3, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 4, 4, 4, 0, 0, 0, 0, 0,
    0, 0, 0, 5, 5, 5, 0, 0, 0, 0,
    0, 0, 0, 0, 6, 6, 6, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0,
    0, 0, 0, 0, 0, 0, 8, 8, 8, 0,
    0, 0, 0, 0, 0, 0, 0, 9, 9, 9,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 12, 10)
  _, n := a.Dims()

  c  := NewFloat64(3.0)
  s  := NewFloat64(2.0)
  t1 := NewFloat64(0.0)
  t2 := NewFloat64(0.0)

  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyTridiagLeft(a1, c, s, i, j, t1, t2)
      ApplyLeft       (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      if i == j {
        continue
      }
      a1 := a.CloneMatrix()
      a2 := a.CloneMatrix()

      ApplyTridiagRight(a1, c, s, i, j, t1, t2)
      ApplyRight       (a2, c, s, i, j, t1, t2)

      if t1.Mnorm(a1.MsubM(a1,a2)).GetFloat64() > 1e-8 {
        test.Errorf("test failed for (i,j) = (%d,%d)", i,j)
      }
    }
  }
}
