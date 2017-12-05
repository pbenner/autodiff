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

import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type NormalIWishartDistribution struct {
  InverseWishartDistribution
  Kappa  Scalar
  Mu     Vector
  r1     Scalar
  r2     Scalar
  sigmap Matrix
}

/* -------------------------------------------------------------------------- */

func NewNormalIWishartDistribution(kappa, nu Scalar, mu Vector, lambda Matrix) (*NormalIWishartDistribution, error) {

  t := kappa.Type()

  n, m := lambda.Dims()

  if n != m || n != mu.Dim() {
    return nil, fmt.Errorf("invalid parameters")
  }

  iw, err := NewInverseWishartDistribution(nu, lambda)
  if err != nil {
    return nil, err
  }

  result := NormalIWishartDistribution{
    InverseWishartDistribution: *iw,
    Kappa : kappa,
    Mu    : mu,
    r1    : NullScalar(t),
    r2    : NullScalar(t),
    sigmap: NullMatrix(t, n, n) }

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

func (dist *NormalIWishartDistribution) ScalarType() ScalarType {
  return dist.Mu.ElementType()
}

func (dist *NormalIWishartDistribution) Dim() int {
  return dist.Mu.Dim()
}

func (dist *NormalIWishartDistribution) MarginalMu() (*TDistribution, error) {
  n := dist.Dim()
  d := NewScalar(dist.ScalarType(), float64(dist.Dim()))
  c := NewScalar(dist.ScalarType(), 0.0)
  c.Add(c.Sub(dist.Nu, d), ConstReal(1.0)) // (nu - d + 1)
  c.Mul(c, dist.Kappa)                   // (nu - d + 1)kappa
  lambda := NullMatrix(dist.ScalarType(), n, n)
  lambda.MdivS(dist.S, c)

  r, err := NewTDistribution(dist.Nu, dist.Mu, lambda)
  if err != nil {
    return nil, err
  }
  return r, nil
}

func (dist *NormalIWishartDistribution) MarginalSigma() (*InverseWishartDistribution, error) {
  r, err := NewInverseWishartDistribution(dist.Nu, dist.S)
  if err != nil {
    return nil, err
  }
  return r, nil
}

func (dist *NormalIWishartDistribution) Mean() (Vector, Matrix, error) {
  t, err := dist.MarginalMu(); if err != nil {
    return nil, nil, err
  }
  w, err := dist.MarginalSigma(); if err != nil {
    return nil, nil, err
  }
  mu1, err := t.Mean(); if err != nil {
    return nil, nil, err
  }
  mu2, err := w.Mean(); if err != nil {
    return nil, nil, err
  }
  return mu1, mu2, nil
}

func (dist *NormalIWishartDistribution) Variance() (Vector, Matrix, error) {
  t, err := dist.MarginalMu(); if err != nil {
    return nil, nil, err
  }
  w, err := dist.MarginalSigma(); if err != nil {
    return nil, nil, err
  }
  var1, err := t.Variance(); if err != nil {
    return nil, nil, err
  }
  var2, err := w.Variance(); if err != nil {
    return nil, nil, err
  }
  return var1, var2, nil
}

func (dist *NormalIWishartDistribution) LogPdf(r Scalar, mu Vector, sigma Matrix) error {
  r1 := dist.r1
  r2 := dist.r2
  sigmap := dist.sigmap
  sigmap.MmulS(sigma, r1.Div(ConstReal(1.0), dist.Kappa))
  if normal, err := NewNormalDistribution(dist.Mu, sigmap); err != nil {
    return err
  } else {
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
