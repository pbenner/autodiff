/* Copyright (C) 2017 Philipp Benner
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

type PowerLawDistribution struct {
  Alpha   Scalar
  Xmin    Scalar
  Epsilon Scalar
  ca      Scalar
  cz      Scalar
}

/* -------------------------------------------------------------------------- */

func NewPowerLawDistribution(alpha, xmin, epsilon Scalar) (*PowerLawDistribution, error) {
  if alpha.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter alpha: %f", alpha.GetValue())
  }
  if xmin.GetValue() == 0.0 {
    return nil, fmt.Errorf("invalid value for parameter x_min: %f", xmin.GetValue())
  }
  c1 := NewBareReal(1.0)
  // some constants
  ca := alpha.Clone()
  ca.Sub(c1, alpha)
  cz := alpha.Clone()
  cz.Sub(alpha, c1)
  cz.Div(cz, xmin)
  cz.Log(cz)

  result := PowerLawDistribution{
    Alpha  : alpha  .Clone(),
    Xmin   : xmin   .Clone(),
    Epsilon: epsilon.Clone(),
    ca     : ca,
    cz     : cz }

  return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *PowerLawDistribution) Clone() *PowerLawDistribution {
  return &PowerLawDistribution{
    Alpha  : dist.Alpha  .Clone(),
    Xmin   : dist.Xmin   .Clone(),
    Epsilon: dist.Epsilon.Clone(),
    ca     : dist.ca     .Clone(),
    cz     : dist.cz     .Clone() }
}

func (dist *PowerLawDistribution) Dim() int {
  return 1
}

func (dist *PowerLawDistribution) LogPdf(x_ Vector) Scalar {
  x := x_[0]
  r := NewScalar(x.Type(), 0.0)
  r.Add(x, dist.Epsilon)

  if r.GetValue() < dist.Xmin.GetValue() {
    r.SetValue(math.Inf(-1))
    return r
  }
  r.Div(r, dist.Xmin)
  r.Log(r)
  r.Mul(r, dist.Alpha)
  r.Neg(r)
  r.Add(r, dist.cz)

  return r
}

func (dist *PowerLawDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *PowerLawDistribution) LogCdf(x_ Vector) Scalar {
  x := x_[0]
  r := NewScalar(x.Type(), 0.0)
  r.Add(x, dist.Epsilon)

  if r.GetValue() <= 0 {
    r.SetValue(math.Inf(-1))
    return r
  }
  r.Div(r, dist.Xmin)
  r.Log(r)
  r.Mul(r, dist.ca)

  return r
}

func (dist *PowerLawDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist PowerLawDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.Alpha
  p[1] = dist.Xmin
  return p
}

func (dist *PowerLawDistribution) SetParameters(parameters Vector) error {
  alpha   := parameters[0]
  xmin    := parameters[1]
  if tmp, err := NewPowerLawDistribution(alpha, xmin, dist.Epsilon); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
