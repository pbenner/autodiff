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
  ca := alpha.CloneScalar()
  ca.Sub(c1, alpha)
  cz := alpha.CloneScalar()
  cz.Sub(alpha, c1)
  cz.Div(cz, xmin)
  cz.Log(cz)

  result := PowerLawDistribution{
    Alpha  : alpha  .CloneScalar(),
    Xmin   : xmin   .CloneScalar(),
    Epsilon: epsilon.CloneScalar(),
    ca     : ca,
    cz     : cz }

  return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *PowerLawDistribution) Clone() *PowerLawDistribution {
  return &PowerLawDistribution{
    Alpha  : dist.Alpha  .CloneScalar(),
    Xmin   : dist.Xmin   .CloneScalar(),
    Epsilon: dist.Epsilon.CloneScalar(),
    ca     : dist.ca     .CloneScalar(),
    cz     : dist.cz     .CloneScalar() }
}

func (dist *PowerLawDistribution) Dim() int {
  return 1
}

func (dist *PowerLawDistribution) ScalarType() ScalarType {
  return dist.Alpha.Type()
}

func (dist *PowerLawDistribution) LogPdf(r Scalar, x Scalar) error {
  r.Add(x, dist.Epsilon)

  if r.GetValue() < dist.Xmin.GetValue() {
    r.SetValue(math.Inf(-1))
    return nil
  }
  r.Div(r, dist.Xmin)
  r.Log(r)
  r.Mul(r, dist.Alpha)
  r.Neg(r)
  r.Add(r, dist.cz)

  return nil
}

func (dist *PowerLawDistribution) Pdf(r Scalar, x Scalar) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *PowerLawDistribution) LogCdf(r Scalar, x Scalar) error {
  r.Add(x, dist.Epsilon)

  if r.GetValue() <= 0 {
    r.SetValue(math.Inf(-1))
    return nil
  }
  r.Div(r, dist.Xmin)
  r.Log(r)
  r.Mul(r, dist.ca)

  return nil
}

func (dist *PowerLawDistribution) Cdf(r Scalar, x Scalar) error {
  if err := dist.LogCdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist PowerLawDistribution) GetParameters() Vector {
  p := NullVector(dist.ScalarType(), 2)
  p.At(0).Set(dist.Alpha)
  p.At(1).Set(dist.Xmin)
  return p
}

func (dist *PowerLawDistribution) SetParameters(parameters Vector) error {
  alpha   := parameters.At(0)
  xmin    := parameters.At(1)
  if tmp, err := NewPowerLawDistribution(alpha, xmin, dist.Epsilon); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
