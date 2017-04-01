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

type GeneralizedGammaDistribution struct {
  A   Scalar
  D   Scalar
  P   Scalar
  dm1 Scalar
  z   Scalar
}

/* -------------------------------------------------------------------------- */

func NewGeneralizedGammaDistribution(a, d, p Scalar) (*GeneralizedGammaDistribution, error) {
  if a.GetValue() <= 0.0 || d.GetValue() <= 0.0 || p.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  dist := GeneralizedGammaDistribution{}
  dist.A = a.Clone()
  dist.D = d.Clone()
  dist.P = p.Clone()
  dist.dm1 = Sub(d, NewScalar(d.Type(), 1.0))
  dist.z   = Log(p)
  dist.z   = Sub(dist.z, Mul(d, Log(a)))
  dist.z   = Sub(dist.z, Lgamma(Div(d, p)))
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) Clone() *GeneralizedGammaDistribution {
  r, _ := NewGeneralizedGammaDistribution(dist.A, dist.D, dist.P)
  return r
}

func (dist *GeneralizedGammaDistribution) Dim() int {
  return 1
}

func (dist *GeneralizedGammaDistribution) LogPdf(x Vector) Scalar {
  if r := x[0].GetValue(); r <= 0.0 || math.IsInf(r, 1) {
    return NewScalar(x.ElementType(), math.Inf(-1))
  }
  t1 := Log(x[0])
  t1.Mul(t1, dist.dm1)
  t2 := Div(x[0], dist.A)
  t2.Pow(t2, dist.P)
  t1.Sub(t1, t2)
  t1.Add(t1, dist.z)
  return t1
}

func (dist *GeneralizedGammaDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *GeneralizedGammaDistribution) LogCdf(x Vector) Scalar {
  panic("not implemented")
}

func (dist *GeneralizedGammaDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) GetParameters() Vector {
  p   := NilVector(3)
  p[0] = dist.A
  p[1] = dist.D
  p[2] = dist.P
  return p
}

func (dist *GeneralizedGammaDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewGeneralizedGammaDistribution(parameters[0], parameters[1], parameters[2]); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
