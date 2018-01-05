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

package scalarDistribution

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type LogCauchyDistribution struct {
  Mu    Scalar
  Sigma Scalar
  s2    Scalar
  z     Scalar
  t     Scalar
}

/* -------------------------------------------------------------------------- */

func NewLogCauchyDistribution(mu, sigma Scalar) (*LogCauchyDistribution, error) {
  if sigma.GetValue() <= 0.0 {
    return nil, fmt.Errorf("invalid parameters")
  }
  t  := mu.Type()
  t1 := NewScalar(t, 0.0)
  t2 := NewScalar(t, 0.0)
  dist := LogCauchyDistribution{}
  dist.Mu    = mu   .CloneScalar()
  dist.Sigma = sigma.CloneScalar()
  dist.z     = t1.Sub(t1.Log(sigma), t2.Log(NewBareReal(math.Pi)))
  dist.s2    = t2.Mul(sigma, sigma)
  return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) Clone() *LogCauchyDistribution {
  r, _ := NewLogCauchyDistribution(dist.Mu, dist.Sigma)
  return r
}

func (dist *LogCauchyDistribution) CloneScalarPdf() ScalarPdf {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) ScalarType() ScalarType {
  return dist.Mu.Type()
}

func (dist *LogCauchyDistribution) LogPdf(r Scalar, x Scalar) error {
  if v := x.GetValue(); v <= 0.0 || math.IsInf(v, 1) {
    r.SetValue(math.Inf(-1))
    return nil
  }
  t := dist.t
  r.Log(x)
  t.Set(r)
  r.Neg(r)
  t.Sub(t, dist.Mu)
  t.Mul(t, t)
  t.Add(t, dist.s2)
  t.Log(t)
  // sum up partial results
  r.Sub(r, t)
  r.Add(r, dist.z)
  return nil
}

func (dist *LogCauchyDistribution) Pdf(r Scalar, x Scalar) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) GetParameters() Vector {
  p := NullVector(dist.ScalarType(), 2)
  p.At(0).Set(dist.Mu)
  p.At(1).Set(dist.Sigma)
  return p
}

func (dist *LogCauchyDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewLogCauchyDistribution(parameters.At(0), parameters.At(1)); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    mu    := NewScalar(t, parameters[0])
    sigma := NewScalar(t, parameters[1])

    if tmp, err := NewLogCauchyDistribution(mu, sigma); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *LogCauchyDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("scalar:log cauchy distribution", dist.GetParameters())
}
