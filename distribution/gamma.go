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
}

/* -------------------------------------------------------------------------- */

func NewGammaDistribution(alpha, beta Scalar) (*GammaDistribution, error) {
  if alpha.GetValue() <= 0.0 || beta.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  dist := GammaDistribution{}
  dist.Alpha = alpha.Clone()
  dist.Beta  = beta .Clone()
  dist.Omega = alpha.Clone()
  dist.Omega.Sub(dist.Omega, NewScalar(alpha.Type(), 1.0))
  dist.Z     = Sub(Mul(alpha, Log(beta)), Lgamma(alpha))
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) Clone() *GammaDistribution {
  r, _ := NewGammaDistribution(dist.Alpha, dist.Beta)
  return r
}

func (dist *GammaDistribution) Dim() int {
  return 1
}

func (dist *GammaDistribution) Mean() Scalar {
  return Div(dist.Alpha, dist.Beta)
}

func (dist *GammaDistribution) LogPdf(x Vector) Scalar {
  if r := x[0].GetValue(); r <= 0.0 || math.IsInf(r, 1) {
    return NewScalar(x.ElementType(), math.Inf(-1))
  }
  t1 := Log(x[0])
  t1.Mul(t1, dist.Omega)
  t2 := Mul(x[0], dist.Beta)
  t1.Sub(t1, t2)
  t1.Add(t1, dist.Z)
  return t1
}

func (dist *GammaDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *GammaDistribution) LogCdf(x Vector) Scalar {
  y := dist.Cdf(x)
  y.Log(y)
  return y
}

func (dist *GammaDistribution) Cdf(x Vector) Scalar {
  y := x[0].Clone()
  y.Mul(y, dist.Beta)
  y.GammaP(dist.Alpha.GetValue(), y)
  return y
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.Alpha
  p[1] = dist.Beta
  return p
}

func (dist *GammaDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewGammaDistribution(parameters[0], parameters[1]); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
