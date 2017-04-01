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

type LogCauchyDistribution struct {
  Mu    Scalar
  Sigma Scalar
  s2    Scalar
  z     Scalar
}

/* -------------------------------------------------------------------------- */

func NewLogCauchyDistribution(mu, sigma Scalar) (*LogCauchyDistribution, error) {
  if sigma.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  dist := LogCauchyDistribution{}
  dist.Mu    = mu   .Clone()
  dist.Sigma = sigma.Clone()
  dist.s2    = Mul(sigma, sigma)
  dist.z     = Sub(Log(sigma), Log(NewBareReal(math.Pi)))
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) Clone() *LogCauchyDistribution {
  r, _ := NewLogCauchyDistribution(dist.Mu, dist.Sigma)
  return r
}

func (dist *LogCauchyDistribution) Dim() int {
  return 1
}

func (dist *LogCauchyDistribution) LogPdf(x Vector) Scalar {
  if r := x[0].GetValue(); r <= 0.0 || math.IsInf(r, 1) {
    return NewScalar(x.ElementType(), math.Inf(-1))
  }
  t1 := Log(x[0])
  t1.Neg(t1)
  t2 := Log(x[0])
  t2.Sub(t2, dist.Mu)
  t2.Mul(t2, t2)
  t2.Add(t2, dist.s2)
  t2.Log(t2)
  // sum up partial results
  t1.Sub(t1, t2)
  t1.Add(t1, dist.z)
  return t1
}

func (dist *LogCauchyDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *LogCauchyDistribution) LogCdf(x Vector) Scalar {
  panic("not implemented")
}

func (dist *LogCauchyDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.Mu
  p[1] = dist.Sigma
  return p
}

func (dist *LogCauchyDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewLogCauchyDistribution(parameters[0], parameters[1]); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
