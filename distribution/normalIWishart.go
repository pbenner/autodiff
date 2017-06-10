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

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type NormalIWishartDistribution struct {
  InverseWishartDistribution
  Kappa  Scalar
  Mu     Vector
  r1     Scalar
  r2     Scalar
}

/* -------------------------------------------------------------------------- */

func NewNormalIWishartDistribution(kappa, nu Scalar, mu Vector, lambda Matrix) (*NormalIWishartDistribution, error) {

  t := kappa.Type()

  n, m := lambda.Dims()

  if n != m || n != mu.Dim() {
    panic("invalid parameters")
  }

  iw, err := NewInverseWishartDistribution(nu, lambda)
  if err != nil {
    return nil, err
  }

  result := NormalIWishartDistribution{
    InverseWishartDistribution: *iw,
    Kappa: kappa,
    Mu   : mu,
    r1   : NewScalar(t, 0.0),
    r2   : NewScalar(t, 0.0) }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *NormalIWishartDistribution) Clone() *NormalIWishartDistribution {
  return &NormalIWishartDistribution{
    InverseWishartDistribution: *dist.InverseWishartDistribution.Clone(),
    Kappa: dist.Kappa.CloneScalar(),
    Mu   : dist.Mu   .CloneVector(),
    r1   : dist.r1   .CloneScalar(),
    r2   : dist.r2   .CloneScalar() }
}

func (dist *NormalIWishartDistribution) Dim() int {
  return dist.Mu.Dim()
}

func (dist *NormalIWishartDistribution) MarginalMu() *TDistribution {
  d := NewReal(float64(dist.Dim()))
  c := Add(Sub(dist.Nu, d), NewReal(1.0)) // (nu - d + 1)
  c  = Mul(dist.Kappa, c)                 // (nu - d + 1)kappa
  lambda := MdivS(dist.S, c)

  r, err := NewTDistribution(dist.Nu, dist.Mu, lambda)
  if err != nil {
    panic(err)
  }
  return r
}

func (dist *NormalIWishartDistribution) MarginalSigma() *InverseWishartDistribution {
  r, err := NewInverseWishartDistribution(dist.Nu, dist.S)
  if err != nil {
    panic(err)
  }
  return r
}

func (dist *NormalIWishartDistribution) Mean() (Vector, Matrix) {
  t := dist.MarginalMu()
  w := dist.MarginalSigma()
  return t.Mean(), w.Mean()
}

func (dist *NormalIWishartDistribution) Variance() (Vector, Matrix) {
  t := dist.MarginalMu()
  w := dist.MarginalSigma()
  return t.Variance(), w.Variance()
}

func (dist *NormalIWishartDistribution) LogPdf(r Scalar, mu Vector, sigma Matrix) error {
  sigmap := MmulS(sigma, Div(NewReal(1.0), dist.Kappa))
  if normal, err := NewNormalDistribution(dist.Mu, sigmap); err != nil {
    return err
  } else {
    r1 := dist.r1
    r2 := dist.r2
    if err := normal.LogPdf(r1, mu); err != nil {
      return err
    }
    if err := dist.InverseWishartDistribution.LogPdf(r2, sigma); err != nil {
      return err
    }
    r.Add(r1, r2)
  }
  return nil
}

func (dist *NormalIWishartDistribution) Pdf(r Scalar, mu Vector, sigma Matrix) error {
  if err := dist.LogPdf(r, mu, sigma); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}
