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

package distribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/determinant"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type InverseWishartDistribution struct {
  Nu   Scalar
  S    Matrix
  SDet Scalar
  n    int    // dimension
}

/* -------------------------------------------------------------------------- */

func NewInverseWishartDistribution(nu Scalar, s Matrix) (*InverseWishartDistribution, error) {

  n, m := s.Dims()

  if n != m {
    panic("NewInverseWishartDistribution(): S is not a square matrix!")
  }
  sDet, err := determinant.Run(s, determinant.PositiveDefinite{true})
  if err != nil {
    return nil, err
  }
  result := InverseWishartDistribution{nu, s, sDet, n}

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *InverseWishartDistribution) Clone() *InverseWishartDistribution {
  return &InverseWishartDistribution{
    Nu  : dist.Nu  .Clone(),
    S   : dist.S   .Clone(),
    SDet: dist.SDet.Clone(),
    n   : dist.n }
}

func (dist *InverseWishartDistribution) Dim() int {
  return dist.n
}

func (dist *InverseWishartDistribution) Mean() Matrix {
  n := dist.Dim()
  if dist.Nu.GetValue() <= float64(n) - 1.0 {
    panic("mean is not defined for the given parameters")
  }
  d := NewReal(float64(n))
  return MdivS(dist.S, Sub(Sub(dist.Nu, d), NewReal(1.0)))
}

func (dist *InverseWishartDistribution) Variance() Matrix {
  n := dist.Dim()
  if dist.Nu.GetValue() <= float64(n) - 1.0 {
    panic("variance is not defined for the given parameters")
  }
  m := NullMatrix(RealType, n, n)
  d := NewReal(float64(n))
  // some constants
  c1 := Sub(dist.Nu, d)                // (nu - d)
  c2 := Add(c1, NewReal(1.0))          // (nu - d + 1)
  c3 := Sub(c1, NewReal(1.0))          // (nu - d - 1)
  c4 := Sub(c1, NewReal(3.0))          // (nu - d - 3)
  c5 := Mul(Mul(c1, Mul(c3, c3)), c4)  // (nu - d)(nu - d - 1)^2 (nu - d - 3)

  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      x := Mul(dist.S.At(i,j), dist.S.At(i,j))
      y := Mul(dist.S.At(i,i), dist.S.At(j,j))
      m.At(i, j).Div(Add(Mul(c2, x), Mul(c3, y)), c5)
    }
  }
  return m
}

func (dist *InverseWishartDistribution) LogPdf(x Matrix) Scalar {
  t := dist.Nu.Type()
  d := NewScalar(t, float64(dist.n))
  xInv, err1 := matrixInverse.Run(x, matrixInverse.PositiveDefinite{true})
  xDet, err2 := determinant  .Run(x, determinant  .PositiveDefinite{true})
  if err1 != nil { panic(err1) }
  if err2 != nil { panic(err2) }
  trace := Mtrace(MmulM(dist.S, xInv))
  c := NewBareReal(2.0)
  // negative log partition function
  z := Mul(Div(dist.Nu, c), Log(dist.SDet))         // |S|^(nu/2)
  z  = Sub(z, Mul(Mul(dist.Nu, Div(d, c)), Log(c))) // 2^(nu n/2)
  z  = Sub(z, Mlgamma(Div(dist.Nu, c), dist.n))     // Gamma_n(nu/2)
  // density
  f := Neg(Mul(Div(Add(Add(dist.Nu, d), NewBareReal(1.0)), c), Log(xDet)))
  f  = Sub(f, Div(trace, c))
  return Add(f, z)
}

func (dist *InverseWishartDistribution) Pdf(x Matrix) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *InverseWishartDistribution) LogCdf(x Matrix) Scalar {
  return Log(dist.Cdf(x))
}

func (dist *InverseWishartDistribution) Cdf(x Matrix) Scalar {
  panic("Not implemented!")
}

/* -------------------------------------------------------------------------- */

func (dist *InverseWishartDistribution) GetParameters() Vector {
  p := dist.S.Vector()
  p  = append(p, dist.Nu)
  return p
}

func (dist *InverseWishartDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  s  := parameters[0:n*n].Matrix(n, n)
  nu := parameters[n*n]
  if tmp, err := NewInverseWishartDistribution(nu, s); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
