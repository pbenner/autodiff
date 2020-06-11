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

package statistics

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "reflect"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ConfigurableDistribution interface {
	ImportConfig(config ConfigDistribution, t ScalarType) error
	ExportConfig() ConfigDistribution
}

/* -------------------------------------------------------------------------- */

type BasicDistribution interface {
	ConfigurableDistribution
	GetParameters() Vector
	SetParameters(parameters Vector) error
	ScalarType() ScalarType
}

/* -------------------------------------------------------------------------- */

type ScalarPdf interface {
	BasicDistribution
	LogPdf(r Scalar, x ConstScalar) error
	CloneScalarPdf() ScalarPdf
}

type VectorPdf interface {
	BasicDistribution
	LogPdf(r Scalar, x ConstVector) error
	Dim() int
	CloneVectorPdf() VectorPdf
}

type MatrixPdf interface {
	BasicDistribution
	LogPdf(r Scalar, x ConstMatrix) error
	Dims() (int, int)
	CloneMatrixPdf() MatrixPdf
}

/* -------------------------------------------------------------------------- */

var ScalarPdfRegistry map[string]ScalarPdf
var VectorPdfRegistry map[string]VectorPdf
var MatrixPdfRegistry map[string]MatrixPdf

func init() {
	ScalarPdfRegistry = make(map[string]ScalarPdf)
	VectorPdfRegistry = make(map[string]VectorPdf)
	MatrixPdfRegistry = make(map[string]MatrixPdf)
}

/* -------------------------------------------------------------------------- */

func NewScalarPdf(name string) ScalarPdf {
	if x, ok := ScalarPdfRegistry[name]; ok {
		return reflect.New(reflect.TypeOf(x).Elem()).Interface().(ScalarPdf)
	} else {
		return nil
	}
}

func NewVectorPdf(name string) VectorPdf {
	if x, ok := VectorPdfRegistry[name]; ok {
		return reflect.New(reflect.TypeOf(x).Elem()).Interface().(VectorPdf)
	} else {
		return nil
	}
}

func NewMatrixPdf(name string) MatrixPdf {
	if x, ok := MatrixPdfRegistry[name]; ok {
		return reflect.New(reflect.TypeOf(x).Elem()).Interface().(MatrixPdf)
	} else {
		return nil
	}
}
