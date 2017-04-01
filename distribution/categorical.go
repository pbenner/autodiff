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

type CategoricalDistribution struct {
  Theta Vector
}

/* -------------------------------------------------------------------------- */

func NewCategoricalDistribution(theta Vector) (*CategoricalDistribution, error) {

  t := theta.Clone()

  for i := 0; i < len(t); i++ {
    if t[i].GetValue() < 0 {
      return nil, fmt.Errorf("invalid negative probability")
    }
    t[i].Log(t[i])
  }
  result := CategoricalDistribution{
    Theta: t }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *CategoricalDistribution) Clone() *CategoricalDistribution {
  return &CategoricalDistribution{
    Theta : dist.Theta.Clone() }
}

func (dist *CategoricalDistribution) Dim() int {
  return 1
}

func (dist *CategoricalDistribution) LogPdf(x Vector) Scalar {
  if len(x) != 1 {
    panic("x has invalid dimension")
  }
  return dist.Theta[int(x[0].GetValue())]
}

func (dist *CategoricalDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *CategoricalDistribution) LogCdf(x Vector) Scalar {
  if len(x) != 1 {
    panic("x has invalid dimension")
  }
  r := NewScalar(x.ElementType(), math.Inf(-1))
  t := NewScalar(x.ElementType(), 0.0)

  for i := 0; i <= int(x[0].GetValue()); i++ {
    r.LogAdd(r, dist.Theta[i], t)
  }
  return r
}

func (dist *CategoricalDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}

/* -------------------------------------------------------------------------- */

func (dist *CategoricalDistribution) GetParameters() Vector {
  return dist.Theta
}

func (dist *CategoricalDistribution) SetParameters(parameters Vector) error {
  dist.Theta = parameters
  return nil
}
