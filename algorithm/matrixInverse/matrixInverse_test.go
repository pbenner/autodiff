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

package matrixInverse

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "testing"
import   "time"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/cholesky"
import   "github.com/pbenner/autodiff/algorithm/gaussJordan"

/* -------------------------------------------------------------------------- */

func TestMatrixInverse(test *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  m2, _ := Run(m1)
  m3 := NewDenseFloat64Matrix([]float64{-2, 1, 1.5, -0.5}, 2, 2)
  t  := NewFloat64(0.0)

  if t.Mnorm(m2.MsubM(m2, m3)).GetFloat64() > 1e-8 {
    test.Error("Inverting matrix failed!")
  }
}

func TestSubmatrixInverse(test *testing.T) {

  // exclude the third row/column
  submatrix := []bool{true, true, false}

  m1 := NewDenseFloat64Matrix([]float64{1,2,50,3,4,60,70,80,90}, 3, 3)
  m2, _ := Run(m1, gaussJordan.Submatrix{submatrix})
  m3 := NewDenseFloat64Matrix([]float64{-2, 1, 0, 1.5, -0.5, 0, 0, 0, 1}, 3, 3)
  t  := NewFloat64(0.0)

  if t.Mnorm(m2.MsubM(m2, m3)).GetFloat64() > 1e-8 {
    test.Error("Inverting matrix failed!")
  }
}

func TestMatrixInversePD(test *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 }, 4, 4)
  m2, _ := Run(m1, PositiveDefinite{true})
  m3 := NewDenseFloat64Matrix([]float64{
     2.515625e+00,  4.843750e-01, -1.296875e+00,  3.593750e-01,
     4.843750e-01,  1.406250e-01, -3.281250e-01,  1.406250e-01,
    -1.296875e+00, -3.281250e-01,  1.015625e+00, -5.781250e-01,
    3.593750e-01,  1.406250e-01, -5.781250e-01,  5.156250e-01 }, 4, 4)
  t  := NewFloat64(0.0)

  if t.Mnorm(m2.MsubM(m2, m3)).GetFloat64() > 1e-8 {
    test.Error("Inverting matrix failed!")
  }
}

func TestMatrixPerformance(test *testing.T) {

  kernelSquaredExponential := func(sigma Matrix, l, v float64) Matrix {
    n, m := sigma.Dims()
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        sigma.At(i, j).SetFloat64(math.Exp(-1.0/2.0*math.Pow(math.Abs(float64(i)-float64(j)), 2.0)/(l*l)))
      }
    }
    return sigma
  }

  n  := 100

  m1 := kernelSquaredExponential(NullDenseReal64Matrix (n, n), 1.0, 1.0)
  m2 := kernelSquaredExponential(NullDenseFloat64Matrix(n, n), 1.0, 1.0)

  // manually initialize matrices
  s1 := InSitu{
    Id: NullDenseReal64Matrix(n, n),
    A : m1,
    B : NullDenseReal64Vector(n),
    Cholesky: cholesky.InSitu{
      L: NullDenseReal64Matrix(n, n),
      D: nil,
      S: NullReal64(),
      T: NullReal64() }}
  s1.Id.SetIdentity()

  s2 := InSitu{
    Id: NullDenseFloat64Matrix(n, n),
    A : m1,
    B : NullDenseFloat64Vector(n),
    Cholesky: cholesky.InSitu{
      L: NullDenseFloat64Matrix(n, n),
      D: nil,
      S: NullFloat64(),
      T: NullFloat64() }}
  s2.Id.SetIdentity()

  start := time.Now()
  Run(m1, PositiveDefinite{true}, &s1)
  elapsed := time.Since(start)
  fmt.Printf("Inverting a 100x100 positive definite matrix (type DenseReal64Matrix) took %s.\n", elapsed)

  start = time.Now()
  Run(m2, PositiveDefinite{true}, &s2)
  elapsed = time.Since(start)
  fmt.Printf("Inverting a 100x100 positive definite matrix (type DenseFloat64Matrix) took %s.\n", elapsed)

}
