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

type BetaDistribution struct {
  Alpha Scalar
  Beta  Scalar
  as1   Scalar
  bs1   Scalar
  z     Scalar
  c1    Scalar
  t1    Scalar
  t2    Scalar
  logScale bool
}

/* -------------------------------------------------------------------------- */

func NewBetaDistribution(alpha, beta Scalar, logScale bool) (*BetaDistribution, error) {
  if alpha.GetValue() <= 0.0 || beta.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  t := alpha.Type()
  dist := BetaDistribution{}
  dist.Alpha = alpha.Clone()
  dist.Beta  = beta .Clone()
  dist.as1 = alpha.Clone()
  dist.bs1 = beta .Clone()
  dist.as1.Sub(alpha, NewBareReal(1.0))
  dist.bs1.Sub(beta,  NewBareReal(1.0))
  dist.logScale = logScale

  t1 := alpha.Clone()
  t1.Add(t1, beta)
  t1.Lgamma(t1)
  t2 := alpha.Clone()
  t2.Lgamma(t2)
  t3 := beta .Clone()
  t3.Lgamma(t3)
  t1.Sub(t1, t2)
  t1.Sub(t1, t3)
  dist.z  = t1
  dist.c1 = NewScalar(t, 1.0)
  dist.t1 = NewScalar(t, 0.0)
  dist.t2 = NewScalar(t, 0.0)
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) Clone() *BetaDistribution {
  r, _ := NewBetaDistribution(dist.Alpha, dist.Beta, dist.logScale)
  return r
}

func (dist *BetaDistribution) Dim() int {
  return 1
}

func (dist *BetaDistribution) ScalarType() ScalarType {
  return dist.Alpha.Type()
}

func (dist *BetaDistribution) LogPdf(r Scalar, x Vector) error {
  if v := x[0].GetValue(); v <= 0.0 || math.IsInf(v, 1) {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t1 := dist.t1
  t2 := dist.t2

  if dist.logScale {
    t1.Mul(x[0], dist.as1)
  } else {
    t1.Log(x[0])
    t1.Mul(t1, dist.as1)
  }
  t2.Sub(dist.c1, x[0])
  t2.Log(t2)
  t2.Mul(t2, dist.bs1)

  r.Add(t1, t2)
  r.Add(r, dist.z)
  
  return nil
}

func (dist *BetaDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) GetParameters() Vector {
  p   := NilVector(2)
  p[0] = dist.Alpha
  p[1] = dist.Beta
  return p
}

func (dist *BetaDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewBetaDistribution(parameters[0], parameters[1], dist.logScale); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
