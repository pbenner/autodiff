/* Copyright (C) 2018 Philipp Benner
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

type PdfLogTransform struct {
  ScalarPdf
  x Scalar
}

/* -------------------------------------------------------------------------- */

func NewPdfLogTransform(scalarPdf ScalarPdf) (*PdfLogTransform, error) {
  r := PdfLogTransform{}
  r.ScalarPdf = scalarPdf
  r.x         = NewScalar(scalarPdf.ScalarType(), 0.0)
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) Clone() *PdfLogTransform {
  r, _ := NewPdfLogTransform(obj.ScalarPdf)
  return r
}

func (obj *PdfLogTransform) CloneScalarPdf() ScalarPdf {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) LogPdf(r Scalar, x Scalar) error {
  if v := x.GetValue(); v < 0.0 {
    r.SetValue(math.Inf(-1))
    return nil
  }
  y := obj.x
  y.Log(x)

  if err := obj.ScalarPdf.LogPdf(r, y); err != nil {
    return err
  }
  r.Sub(r, y)

  return nil
}

func (obj *PdfLogTransform) Pdf(r Scalar, x Scalar) error {
  if err := obj.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if len(config.Distributions) != 1 {
    return fmt.Errorf("invalid config file")
  }
  if tmp, err := ImportScalarPdfConfig(config.Distributions[0], t); err != nil {
    return err
  } else {
    if tmp, err := NewPdfLogTransform(tmp); err != nil {
      return err
    } else {
      *obj = *tmp
    }
  }
  return nil
}

func (obj *PdfLogTransform) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("scalar:log pdf transform", obj.GetParameters())
}
