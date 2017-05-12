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

type NegativeBinomialDistribution struct {
  R     Scalar
  P     Scalar
  q     Scalar // q = log(1-p)
  z     Scalar
  c1    Scalar
  t1    Scalar
  t2    Scalar
}

/* -------------------------------------------------------------------------- */

func NewNegativeBinomialDistribution(r, p Scalar) (*NegativeBinomialDistribution, error) {
  if r.GetValue() <= 0.0 || p.GetValue() < 0.0 || p.GetValue() > 1.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  t := r.Type()

  // p^r
  t1 := p.Clone()
  t1.Log(t1)
  t1.Mul(t1, r)

  // Gamma(r)
  t2 := r.Clone()
  t2.Lgamma(t2)

  // p^r / Gamma(r)
  t1.Sub(t1, t2)
  
  dist := NegativeBinomialDistribution{}
  dist.R  = r.Clone()
  dist.P  = p.Clone()
  dist.q  = Log(Sub(NewReal(1.0), p))
  dist.z  = t1
  dist.c1 = NewScalar(t, 1.0)
  dist.t1 = NewScalar(t, 0.0)
  dist.t2 = NewScalar(t, 0.0)
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) Clone() *NegativeBinomialDistribution {
  r, _ := NewNegativeBinomialDistribution(dist.R, dist.P)
  return r
}

func (dist *NegativeBinomialDistribution) Dim() int {
  return 1
}

func (dist *NegativeBinomialDistribution) ScalarType() ScalarType {
  return dist.R.Type()
}

func (dist *NegativeBinomialDistribution) LogPdf(r Scalar, x Vector) error {
  if v := x[0].GetValue(); v < 0.0 || math.Floor(v) != v {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t1 := dist.t1
  t2 := dist.t2

  // Gamma(r + k)
  t1.Add(dist.R, x[0])
  t1.Lgamma(t1)

  // Gamma(k - 1)
  t2.Sub(x[0], dist.c1)
  t2.Lgamma(t2)

  r.Mul(x[0], dist.q)
  r.Add(r, t1)
  r.Sub(r, t2)
  r.Add(r, dist.z)

  return nil
}

func (dist *NegativeBinomialDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.R
  p[1] = dist.P
  return p
}

func (dist *NegativeBinomialDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewNegativeBinomialDistribution(parameters[0], parameters[1]); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
