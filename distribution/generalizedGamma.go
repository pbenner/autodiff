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
  t   Scalar
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
  dist.t   = NewScalar(a.Type(), 0.0)
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

func (dist *GeneralizedGammaDistribution) ScalarType() ScalarType {
  return dist.A.Type()
}

func (dist *GeneralizedGammaDistribution) LogPdf(r Scalar, x Vector) error {
  if v := x[0].GetValue(); v <= 0.0 || math.IsInf(v, 1) {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t := dist.t
  t.Div(x[0], dist.A)
  t.Pow(t, dist.P)

  r.Log(x[0])
  r.Mul(r, dist.dm1)
  r.Sub(r, t)
  r.Add(r, dist.z)
  return nil
}

func (dist *GeneralizedGammaDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
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
