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

type ParetoDistribution struct {
  Lambda    Scalar // scale
  Kappa     Scalar // shape
  Kappa1p   Scalar
  Epsilon   Scalar // x' = x + epsilon
  z         Scalar
}

/* -------------------------------------------------------------------------- */

func NewParetoDistribution(lambda, kappa, epsilon Scalar) (*ParetoDistribution, error) {
  if lambda.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter lambda: %f", lambda.GetValue())
  }
  if kappa.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter kappa: %f", kappa.GetValue())
  }
  if epsilon.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter epsilon: %f", kappa.GetValue())
  }

  kappa1p  := kappa.Clone()
  kappa1p.Add(kappa1p, NewBareReal(1.0))

  z := Add(Log(kappa), Mul(kappa, Log(lambda)))

  result := &ParetoDistribution{
    Lambda   : lambda .Clone(),
    Kappa    : kappa  .Clone(),
    Kappa1p  : kappa1p.Clone(),
    Epsilon  : epsilon.Clone(),
    z        : z }

  return result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) Clone() *ParetoDistribution {
  return &ParetoDistribution{
    Lambda   : dist.Lambda .Clone(),
    Kappa    : dist.Kappa  .Clone(),
    Kappa1p  : dist.Kappa1p.Clone(),
    Epsilon  : dist.Epsilon.Clone(),
    z        : dist.z      .Clone()}
}

func (dist *ParetoDistribution) Dim() int {
  return 1
}

func (dist *ParetoDistribution) LogPdf(r Scalar, x_ Vector) error {
  x := x_[0]

  if x.GetValue() < 0 {
    r.SetValue(math.Inf(-1))
    return nil
  }

  r.Add(x, dist.Epsilon)
  r.Log(r)
  r.Mul(r, dist.Kappa1p)
  r.Neg(r)
  r.Add(r, dist.z)

  return nil
}

func (dist *ParetoDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *ParetoDistribution) LogCdf(r Scalar, x_ Vector) error {
  x := x_[0]

  if x.GetValue() < 0 {
    r.SetValue(math.Inf(-1))
    return nil
  }

  r.Add(x, dist.Epsilon)
  r.Div(dist.Lambda, r)
  r.Pow(r, dist.Kappa)
  r.Neg(r)
  r.Log1p(r)

  return nil
}

func (dist *ParetoDistribution) Cdf(r Scalar, x Vector) error {
  if err := dist.LogCdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist ParetoDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.Lambda
  p[1] = dist.Kappa
  return p
}

func (dist *ParetoDistribution) SetParameters(parameters Vector) error {
  lambda  := parameters[0]
  kappa   := parameters[1]
  if tmp, err := NewParetoDistribution(lambda, kappa, dist.Epsilon); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
