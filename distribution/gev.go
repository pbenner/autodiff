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

type GevDistribution struct {
  Mu    Scalar
  Sigma Scalar
  Xi    Scalar
  c1    Scalar
  cx    Scalar
  cy    Scalar
  t     Scalar
}

/* -------------------------------------------------------------------------- */

func NewGevDistribution(mu, sigma, xi Scalar) (*GevDistribution, error) {
  if sigma.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid value for parameter sigma: %f", sigma.GetValue())
  }
  // some constants
  c1 := NewBareReal(1.0)
  cx := NewScalar(xi.Type(), 1.0)
  cx.Div(cx, xi)
  cx.Neg(cx)
  cy := NewScalar(xi.Type(), 1.0)
  cy.Div(cy, xi)
  cy.Add(cy, c1)

  result := GevDistribution{
    Mu    : mu   .Clone(),
    Sigma : sigma.Clone(),
    Xi    : xi   .Clone(),
    c1    : c1,
    cx    : cx,
    cy    : cy,
    t     : NewScalar(xi.Type(), 0.0) }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) Clone() *GevDistribution {
  return &GevDistribution{
    Mu    : dist.Mu   .Clone(),
    Sigma : dist.Sigma.Clone(),
    Xi    : dist.Xi   .Clone(),
    c1    : dist.c1   .Clone(),
    cx    : dist.cx   .Clone(),
    cy    : dist.cy   .Clone(),
    t     : dist.t    .Clone() }
}

func (dist *GevDistribution) Dim() int {
  return 1
}

func (dist *GevDistribution) ScalarType() ScalarType {
  return dist.Mu.Type()
}

func (dist *GevDistribution) LogPdf(r Scalar, x_ Vector) error {
  x := x_[0]

  if dist.Xi.GetValue()*(x.GetValue() - dist.Mu.GetValue())/dist.Sigma.GetValue() <= -1 {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t := dist.t
  t.Sub(x, dist.Mu)
  t.Div(t, dist.Sigma)

  if dist.Xi.GetValue() == 0.0 {
    t.Neg(t)
    // r = - (x-mu)/sigma
    r.Set(t)
    t.Exp(t)
    // r = - (x-mu)/sigma - exp{-(x-mu)/sigma}
    r.Sub(r, t)

  } else {
    t.Mul(t, dist.Xi)
    t.Add(t, dist.c1)
    r.Pow(t, dist.cx)
    // r = - (1 + xi(x-mu)/sigma)^(-1/xi)
    r.Neg(r)

    t.Log(t)
    t.Mul(t, dist.cy)
    // r = - (1+1/xi) log(1 + xi(x-mu)/sigma) - (1 + xi(x-mu)/sigma)^(-1/xi)
    r.Sub(r, t)
  }
  t.Log(dist.Sigma)
  r.Sub(r, t)

  return nil
}

func (dist *GevDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *GevDistribution) LogCdf(r Scalar, x_ Vector) error {
  x := x_[0]

  if dist.Xi.GetValue()*(x.GetValue() - dist.Mu.GetValue())/dist.Sigma.GetValue() <= -1 {
    r.SetValue(math.Inf(-1))
    return nil
  }
  r.Set(x)
  r.Sub(r, dist.Mu)
  r.Div(r, dist.Sigma)

  if dist.Xi.GetValue() == 0.0 {
    r.Neg(r)
    r.Exp(r)
    r.Neg(r)
  } else {
    r.Mul(r, dist.Xi)
    r.Add(r, dist.c1)
    r.Pow(r, dist.cx)
    r.Neg(r)
  }

  return nil
}

func (dist *GevDistribution) Cdf(r Scalar, x Vector) error {
  if err := dist.LogCdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) GetParameters() Vector {
  p := NilVector(3)
  p[0] = dist.Mu
  p[1] = dist.Sigma
  p[2] = dist.Xi
  return p
}

func (dist *GevDistribution) SetParameters(parameters Vector) error {
  mu    := parameters[0]
  sigma := parameters[1]
  xi    := parameters[2]
  if tmp, err := NewGevDistribution(mu, sigma, xi); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
