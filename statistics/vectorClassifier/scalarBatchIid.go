/* Copyright (C) 2017 Philipp Benner
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

package vectorClassifier

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type ScalarBatchIid struct {
	Classifier ScalarBatchClassifier
	N          int
}

/* -------------------------------------------------------------------------- */

func NewScalarBatchIid(classifier ScalarBatchClassifier, n int) (ScalarBatchIid, error) {
	if n < -1 && n != 0 {
		return ScalarBatchIid{}, fmt.Errorf("invalid dimension")
	}
	return ScalarBatchIid{classifier, n}, nil
}

/* -------------------------------------------------------------------------- */

func (obj ScalarBatchIid) CloneVectorBatchClassifier() VectorBatchClassifier {
	return ScalarBatchIid{obj.Classifier.CloneScalarBatchClassifier(), obj.N}
}

/* -------------------------------------------------------------------------- */

func (obj ScalarBatchIid) Dim() int {
	return obj.N
}

func (obj ScalarBatchIid) Eval(r Scalar, x ConstVector) error {
	if obj.N != -1 && obj.N != x.Dim() {
		return fmt.Errorf("data has invalid dimension (expected dimension `%d' but data has dimension `%d')", obj.N, x.Dim())
	}
	if x.Dim() == 1 {
		return obj.Classifier.Eval(r, x.ConstAt(0))
	} else {
		t := r.CloneScalar()
		r.SetValue(math.Inf(-1))
		for i := 0; i < x.Dim(); i++ {
			if err := obj.Classifier.Eval(t, x.ConstAt(i)); err != nil {
				return err
			}
			r.Add(r, t)
		}
	}
	return nil
}
