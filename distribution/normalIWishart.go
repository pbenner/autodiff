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
}

/* -------------------------------------------------------------------------- */

func NewNormalIWishartDistribution(kappa, nu Scalar, mu Vector, lambda Matrix) (*NormalIWishartDistribution, error) {

  n, m := lambda.Dims()

  if n != m || n != len(mu) {
    panic("invalid parameters")
  }

  iw, err := NewInverseWishartDistribution(nu, lambda)
  if err != nil {
    return nil, err
  }

  result := NormalIWishartDistribution{*iw, kappa, mu}

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *NormalIWishartDistribution) Clone() *NormalIWishartDistribution {
  return &NormalIWishartDistribution{
    *dist.InverseWishartDistribution.Clone(),
    dist.Kappa.Clone(), dist.Mu.Clone()}
}

func (dist *NormalIWishartDistribution) Dim() int {
  return len(dist.Mu)
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

func (dist *NormalIWishartDistribution) LogPdf(mu Vector, sigma Matrix) Scalar {
  sigmap := MmulS(sigma, Div(NewReal(1.0), dist.Kappa))
  normal, err := NewNormalDistribution(dist.Mu, sigmap)
  if err != nil {
    panic(err)
  }
  return Add(normal.LogPdf(mu), dist.InverseWishartDistribution.LogPdf(sigma))
}

func (dist *NormalIWishartDistribution) Pdf(mu Vector, sigma Matrix) Scalar {
  return Exp(dist.LogPdf(mu, sigma))
}

func (dist *NormalIWishartDistribution) LogCdf(x Vector) Scalar {
  return Log(dist.Cdf(x))
}

func (dist *NormalIWishartDistribution) Cdf(x Vector) Scalar {
  panic("not implemented")
}
