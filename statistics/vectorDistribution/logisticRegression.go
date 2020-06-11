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

import "fmt"

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
	r.Theta = theta
	r.t = NullScalar(theta.ElementType())
	return &r, nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogisticRegression) Clone() *LogisticRegression {
	return &LogisticRegression{
		Theta: dist.Theta.CloneVector(),
		t:     dist.t.CloneScalar()}
}

func (obj *LogisticRegression) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *LogisticRegression) Dim() int {
	return dist.Theta.Dim() - 1
}

func (dist *LogisticRegression) ScalarType() ScalarType {
	return dist.Theta.ElementType()
}

func (dist *LogisticRegression) ClassLogPdf(r Scalar, x ConstVector, y bool) error {
	if x.Dim() != dist.Dim() && x.Dim() != dist.Dim()+1 {
		return fmt.Errorf("input vector has invalid dimension")
	}
	t := dist.t
	// set r to first element of theta
	r.Set(dist.Theta.ConstAt(0))
	// loop over theta
	it := x.ConstIterator()
	if x.Dim() == dist.Dim() {
		// dim(x) == dim(theta)-1 => add 1 to index
		for ; it.Ok(); it.Next() {
			t.Mul(it.GetConst(), dist.Theta.ConstAt(it.Index()+1))
			r.Add(r, t)
		}
	} else {
		// dim(x) == dim(theta) => ignore first element of x
		if it.Ok() {
			it.Next()
		}
		for ; it.Ok(); it.Next() {
			t.Mul(it.GetConst(), dist.Theta.ConstAt(it.Index()+0))
			r.Add(r, t)
		}
	}
	if y {
		r.Neg(r)
		r.LogAdd(ConstReal(0.0), r, t)
	} else {
		r.LogAdd(ConstReal(0.0), r, t)
	}
	r.Neg(r)
	return nil
}

func (dist *LogisticRegression) ClassLogPdf_(r *BareReal, x SparseConstRealVector, y bool) error {
	theta := dist.Theta.(DenseBareRealVector)
	if x.Dim() != dist.Dim() && x.Dim() != dist.Dim()+1 {
		return fmt.Errorf("input vector has invalid dimension")
	}
	t := BareReal(0.0)
	// set r to first element of theta
	r.Set(theta.ConstAt(0))
	// loop over theta
	it := x.ITERATOR()
	if x.Dim() == dist.Dim() {
		// dim(x) == dim(theta)-1 => add 1 to index
		for ; it.Ok(); it.Next() {
			y := BareReal(it.GET())
			t.MUL(&y, theta.AT(it.Index()+1))
			r.ADD(r, &t)
		}
	} else {
		// dim(x) == dim(theta) => ignore first element of x
		if it.Ok() {
			it.Next()
		}
		for ; it.Ok(); it.Next() {
			y := BareReal(it.GET())
			t.MUL(&y, theta.AT(it.Index()+0))
			r.ADD(r, &t)
		}
	}
	if y {
		r.NEG(r)
		r.LogAdd(ConstReal(0.0), r, &t)
	} else {
		r.LogAdd(ConstReal(0.0), r, &t)
	}
	r.NEG(r)
	return nil
}

func (dist *LogisticRegression) LogPdf(r Scalar, x ConstVector) error {
	return dist.ClassLogPdf(r, x, true)
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

	theta, ok := config.GetNamedParametersAsVector("Theta", t)
	if !ok {
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

	config := struct {
		Theta []float64
	}{}
	config.Theta = obj.Theta.GetValues()

	return NewConfigDistribution("vector:logistic regression", config)
}
