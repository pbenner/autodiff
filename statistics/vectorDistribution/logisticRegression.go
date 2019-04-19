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

package vectorDistribution

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type LogisticRegression struct {
  Theta Vector
  t     Scalar
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(theta Vector) (*LogisticRegression, error) {
  r := LogisticRegression{}
  r.Theta = theta.CloneVector()
  r.t     = NullScalar(theta.ElementType())
  return &r, nil

}

/* -------------------------------------------------------------------------- */

func (dist *LogisticRegression) Clone() *LogisticRegression {
  return &LogisticRegression{
    Theta : dist.Theta.CloneVector(),
    t     : dist.t    .CloneScalar() }
}

func (obj *LogisticRegression) CloneVectorPdf() VectorPdf {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *LogisticRegression) Dim() int {
  return dist.Theta.Dim()-1
}

func (dist *LogisticRegression) ScalarType() ScalarType {
  return dist.Theta.ElementType()
}

func (dist *LogisticRegression) ClassLogPdf(r Scalar, x ConstVector, y bool) error {
  if x.Dim() != dist.Dim() {
    return fmt.Errorf("input vector has invalid dimension")
  }
  t := dist.t
  r.Set(dist.Theta.ConstAt(0))
  for i := 0; i < dist.Dim(); i++ {
    t.Mul(dist.Theta.ConstAt(i+1), x.ConstAt(i))
    r.Add(r, t)
  }
  if y {
    r.Neg(r)
    r.LogAdd(ConstReal(0.0), r, t)
    r.Neg(r)
  } else {
    r.LogAdd(ConstReal(0.0), r, t)
    r.Neg(r)
  }
  return nil
}

func (dist *LogisticRegression) LogPdf(r Scalar, x ConstVector) error {
  if x.Dim() != dist.Dim() {
    return fmt.Errorf("input vector has invalid dimension")
  }
  t := dist.t
  r.Set(dist.Theta.ConstAt(0))
  for i := 0; i < dist.Dim(); i++ {
    t.Mul(dist.Theta.ConstAt(i+1), x.ConstAt(i))
    r.Add(r, t)
  }
  r.Neg(r)
  r.LogAdd(ConstReal(0.0), r, t)
  r.Neg(r)
  return nil
}

func (dist *LogisticRegression) Pdf(r Scalar, x ConstVector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogisticRegression) GetParameters() Vector {
  return dist.Theta
}

func (dist *LogisticRegression) SetParameters(parameters Vector) error {
  if parameters.Dim() != dist.Dim()+1 {
    return fmt.Errorf("invalid number of parameters for logistic regression model")
  }
  dist.Theta = parameters
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) ImportConfig(config ConfigDistribution, t ScalarType) error {

  theta, ok := config.GetNamedParametersAsVector("Theta", t); if !ok {
    return fmt.Errorf("invalid config file")
  }
  if tmp, err := NewLogisticRegression(theta); err != nil {
    return err
  } else {
    *obj = *tmp
  }
  return nil
}

func (obj *LogisticRegression) ExportConfig() ConfigDistribution {

  config := struct{
    Theta []float64
  }{}
  config.Theta = obj.Theta.GetValues()

  return NewConfigDistribution("vector:logistic regression", config)
}
