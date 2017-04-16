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
  d    Scalar
  z    Scalar
  c1   Scalar
  c2   Scalar
  // state
  t    Scalar
  inSituDet   determinant.InSitu
  inSituInv matrixInverse.InSitu
}

/* -------------------------------------------------------------------------- */

func NewInverseWishartDistribution(nu Scalar, s Matrix) (*InverseWishartDistribution, error) {

  t := nu.Type()

  n, m := s.Dims()

  if n != m {
    panic("NewInverseWishartDistribution(): S is not a square matrix!")
  }
  sDet, err := determinant.Run(s, determinant.PositiveDefinite{true})
  if err != nil {
    return nil, err
  }
  d  := NewScalar(t, float64(n))
  c1 := NewBareReal(1.0)
  c2 := NewBareReal(2.0)
  // negative log partition function
  z := Mul(Div(nu, c2), Log(sDet))            // |S|^(nu/2)
  z.Sub(z, Mul(Mul(nu, Div(d, c2)), Log(c2))) // 2^(nu n/2)
  z.Sub(z, Mlgamma(Div(nu, c2), n))           // Gamma_n(nu/2)

  result := InverseWishartDistribution{
    Nu  : nu,
    S   : s,
    SDet: sDet,
    d   : d,
    c1  : c1,
    c2  : c2,
    z   : z,
    t   : NewScalar(t, 0.0) }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *InverseWishartDistribution) Clone() *InverseWishartDistribution {
  return &InverseWishartDistribution{
    Nu  : dist.Nu  .Clone(),
    S   : dist.S   .Clone(),
    SDet: dist.SDet.Clone(),
    d   : dist.d   .Clone(),
    c1  : dist.c1  .Clone(),
    c2  : dist.c2  .Clone(),
    z   : dist.z   .Clone(),
    t   : dist.t   .Clone() }
}

func (dist *InverseWishartDistribution) Dim() int {
  n, _ := dist.S.Dims()
  return n
}

func (dist *InverseWishartDistribution) Mean() Matrix {
  n := dist.Dim()
  if dist.Nu.GetValue() <= float64(n) - 1.0 {
    panic("mean is not defined for the given parameters")
  }
  return MdivS(dist.S, Sub(Sub(dist.Nu, dist.d), NewReal(1.0)))
}

func (dist *InverseWishartDistribution) Variance() Matrix {
  n := dist.Dim()
  if dist.Nu.GetValue() <= float64(n) - 1.0 {
    panic("variance is not defined for the given parameters")
  }
  m := NullMatrix(RealType, n, n)
  // some constants
  c1 := Sub(dist.Nu, dist.d)           // (nu - d)
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

func (dist *InverseWishartDistribution) LogPdf(r Scalar, x Matrix) error {
  xInv, err1 := matrixInverse.Run(x, matrixInverse.PositiveDefinite{true}, &dist.inSituInv)
  xDet, err2 := determinant  .Run(x, determinant  .PositiveDefinite{true}, &dist.inSituDet)
  if err1 != nil { return err1 }
  if err2 != nil { return err2 }
  xDet.Log(xDet)
  xInv.MmulM(dist.S, xInv)
  t := dist.t
  t.Mtrace(xInv)
  t.Div(t, dist.c2)
  // density
  r.Add(dist.Nu, dist.d)
  r.Add(r, dist.c1)
  r.Div(r, dist.c2)
  r.Mul(r, xDet)
  r.Neg(r)
  r.Sub(r, t)
  r.Add(r, dist.z)
  return nil
}

func (dist *InverseWishartDistribution) Pdf(r Scalar, x Matrix) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
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
