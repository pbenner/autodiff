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

type ExponentialDistribution struct {
  Lambda    Scalar
  LambdaLog Scalar
  c1        Scalar
}

/* -------------------------------------------------------------------------- */

func NewExponentialDistribution(lambda Scalar) (*ExponentialDistribution, error) {
  if lambda.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter lambda: %f", lambda.GetValue())
  }
  // some constants
  c1 := NewBareReal(1.0)

  result := ExponentialDistribution{
    Lambda   : lambda.Clone(),
    LambdaLog: Log(lambda),
    c1       : c1 }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) Clone() *ExponentialDistribution {
  return &ExponentialDistribution{
    Lambda   : dist.Lambda   .Clone(),
    LambdaLog: dist.LambdaLog.Clone(),
    c1       : dist.c1       .Clone() }
}

func (dist *ExponentialDistribution) Dim() int {
  return 1
}

func (dist *ExponentialDistribution) LogPdf(x_ Vector) Scalar {
  x := x_[0]
  r := NewScalar(x.Type(), 0.0)

  if x.GetValue() < 0 {
    r.SetValue(math.Inf(-1))
    return r
  }

  r.Mul(dist.Lambda, x)
  r.Neg(r)
  r.Add(r, dist.LambdaLog)

  return r
}

func (dist *ExponentialDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *ExponentialDistribution) LogCdf(x_ Vector) Scalar {
  x := x_[0]
  r := NewScalar(x.Type(), 0.0)

  if x.GetValue() < 0 {
    r.SetValue(math.Inf(-1))
    return r
  }

  r.Mul(dist.Lambda, x)
  r.Neg(r)
  r.Exp(r)
  r.Neg(r)
  r.Log1p(r)

  return r
}

func (dist *ExponentialDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist ExponentialDistribution) GetParameters() Vector {
  p   := NilVector(1)
  p[0] = dist.Lambda
  return p
}

func (dist *ExponentialDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewExponentialDistribution(parameters[0]); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
