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

/* -------------------------------------------------------------------------- */

type ChiSquaredDistribution struct {
  K Scalar
  C Scalar // c = 2
  L Scalar // l = k/2
  E Scalar // k/2 - 1
  Z Scalar // k/2 log 2 + log Gamma(k/2)
}

/* -------------------------------------------------------------------------- */

func NewChiSquaredDistribution(k_ float64) (*ChiSquaredDistribution, error) {
  // we cannot differentiate with respect to k, so use bare reals
  t  := BareRealType
  k  := NewScalar(t, k_)
  c1 := NewScalar(t, 1.0)
  c2 := NewScalar(t, 2.0)
  l  := Div(k, c2)
  e  := Sub(l, c1)
  z  := Add(Mul(l, Log(c2)), Lgamma(l))
  return &ChiSquaredDistribution{K: k, C: c2, L: l, E: e, Z: z}, nil
}

/* -------------------------------------------------------------------------- */

func (dist *ChiSquaredDistribution) Clone() *ChiSquaredDistribution {
  r, _ := NewChiSquaredDistribution(dist.K.GetValue())
  return r
}

func (dist *ChiSquaredDistribution) Dim() int {
  return 1
}

func (dist *ChiSquaredDistribution) LogPdf(r Scalar, x Vector) error {
  r.Log(x.At(0))
  r.Mul(r, dist.E)
  t := Div(x.At(0), dist.C)
  r.Sub(r, t)
  r.Sub(r, dist.Z)
  return nil
}

func (dist *ChiSquaredDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *ChiSquaredDistribution) LogCdf(r Scalar, x Vector) error {
  if err := dist.Cdf(r, x); err != nil {
    return err
  }
  r.Log(r)
  return nil
}

func (dist *ChiSquaredDistribution) Cdf(r Scalar, x Vector) error {
  r.Div(x.At(0), dist.C)
  r.GammaP(dist.L.GetValue(), r)
  return nil
}
