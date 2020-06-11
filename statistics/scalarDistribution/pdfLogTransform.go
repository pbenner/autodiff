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

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type PdfLogTransform struct {
	ScalarPdf
	c float64
	x Scalar
}

/* -------------------------------------------------------------------------- */

func NewPdfLogTransform(scalarPdf ScalarPdf, pseudocount float64) (*PdfLogTransform, error) {
	r := PdfLogTransform{}
	r.ScalarPdf = scalarPdf
	r.c = pseudocount
	r.x = NewScalar(scalarPdf.ScalarType(), 0.0)
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) Clone() *PdfLogTransform {
	r, _ := NewPdfLogTransform(obj.ScalarPdf.CloneScalarPdf(), obj.c)
	return r
}

func (obj *PdfLogTransform) CloneScalarPdf() ScalarPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); v < 0.0 {
		r.SetValue(math.Inf(-1))
		return nil
	}
	y := obj.x
	y.Add(x, ConstReal(obj.c))
	y.Log(y)

	if err := obj.ScalarPdf.LogPdf(r, y); err != nil {
		return err
	}
	r.Sub(r, y)

	return nil
}

func (obj *PdfLogTransform) Pdf(r Scalar, x ConstScalar) error {
	if err := obj.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *PdfLogTransform) ImportConfig(config ConfigDistribution, t ScalarType) error {

	parameters, ok := config.GetParametersAsFloats()
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	if len(parameters) != 1 {
		return fmt.Errorf("invalid config file")
	}

	if len(config.Distributions) != 1 {
		return fmt.Errorf("invalid config file")
	}
	if tmp, err := ImportScalarPdfConfig(config.Distributions[0], t); err != nil {
		return err
	} else {
		if tmp, err := NewPdfLogTransform(tmp, parameters[0]); err != nil {
			return err
		} else {
			*obj = *tmp
		}
	}
	return nil
}

func (obj *PdfLogTransform) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:pdf log transform", []float64{obj.c}, obj.ScalarPdf.ExportConfig())
}
