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
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type GammaDistribution struct {
  Alpha Scalar // shape
  Beta  Scalar // rate
  Omega Scalar
  Z     Scalar
  t     Scalar
}

/* -------------------------------------------------------------------------- */

func NewGammaDistribution(alpha, beta Scalar) (*GammaDistribution, error) {
  if alpha.GetValue() <= 0.0 || beta.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  t := alpha.Type()
  dist := GammaDistribution{}
  dist.Alpha = alpha.CloneScalar()
  dist.Beta  = beta .CloneScalar()
  dist.Omega = alpha.CloneScalar()
  dist.Omega.Sub(dist.Omega, NewScalar(t, 1.0))
  dist.Z     = Sub(Mul(alpha, Log(beta)), Lgamma(alpha))
  dist.t     = NewScalar(t, 0.0)
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) CloneScalar() *GammaDistribution {
  r, _ := NewGammaDistribution(dist.Alpha, dist.Beta)
  return r
}

func (dist *GammaDistribution) Dim() int {
  return 1
}

func (dist *GammaDistribution) ScalarType() ScalarType {
  return dist.Alpha.Type()
}

func (dist *GammaDistribution) Mean() Scalar {
  return Div(dist.Alpha, dist.Beta)
}

func (dist *GammaDistribution) LogPdf(r Scalar, x Vector) error {
  if v := x.At(0).GetValue(); v <= 0.0 || math.IsInf(v, 1) {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t := dist.t
  t.Mul(x.At(0), dist.Beta)

  r.Log(x.At(0))
  r.Mul(r, dist.Omega)
  r.Sub(r, t)
  r.Add(r, dist.Z)
  return nil
}

func (dist *GammaDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *GammaDistribution) LogCdf(r Scalar, x Vector) error {
  if err := dist.Cdf(r, x); err != nil {
    return err
  }
  r.Log(r)
  return nil
}

func (dist *GammaDistribution) Cdf(r Scalar, x Vector) error {
  r.Mul(x.At(0), dist.Beta)
  r.GammaP(dist.Alpha.GetValue(), r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) GetParameters() Vector {
  p   := NilDenseVector(2)
  p[0] = dist.Alpha
  p[1] = dist.Beta
  return p
}

func (dist *GammaDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewGammaDistribution(parameters.At(0), parameters.At(1)); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
