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

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type LaplaceDistribution struct {
  Mu    Scalar
  Sigma Scalar
}

/* -------------------------------------------------------------------------- */

func NewLaplaceDistribution(mu, sigma Scalar) (*LaplaceDistribution, error) {

  result := LaplaceDistribution{}
  result.Mu    = mu   .Clone()
  result.Sigma = sigma.Clone()

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *LaplaceDistribution) Clone() *LaplaceDistribution {
  return &LaplaceDistribution{
    Mu      : dist.Mu.Clone(),
    Sigma   : dist.Sigma.Clone() }
}

func (dist *LaplaceDistribution) Dim() int {
  return 1
}

func (dist *LaplaceDistribution) LogPdf(x Vector) Scalar {

  t := x.ElementType()
  c := NewScalar(t, 2.0)

  r := Sub(x[0], dist.Mu)
  r.Abs(r)
  r.Div(r, dist.Sigma)
  r.Neg(r)
  r.Exp(r)
  r.Div(r, dist.Sigma)
  r.Div(r, c)
  
  return r
}

func (dist *LaplaceDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *LaplaceDistribution) LogCdf(x Vector) Scalar {

  t  := x.ElementType()
  c1 := NewScalar(t, 1.0)
  c2 := NewScalar(t, 2.0)

  r := Sub(x[0], dist.Mu)
  r.Abs(r)
  r.Div(r, dist.Sigma)
  r.Neg(r)
  r.Exp(r)
  r.Div(r, c2)

  if x[0].Greater(dist.Mu) {
    r.Neg(r)
    r.Add(r, c1)
  }
  return r
}

func (dist *LaplaceDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist *LaplaceDistribution) GetParameters() Vector {
  p := NilVector(2)
  p[0] = dist.Mu
  p[1] = dist.Sigma
  return p
}

func (dist *LaplaceDistribution) SetParameters(parameters Vector) error {
  mu    := parameters[0]
  sigma := parameters[1]
  if tmp, err := NewLaplaceDistribution(mu, sigma); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
