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
import . "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type HmmPosterior struct {
	*Hmm
	States []int
}

/* -------------------------------------------------------------------------- */

func (obj HmmPosterior) CloneVectorClassifier() VectorClassifier {
	return HmmPosterior{obj.Clone(), obj.States}
}

/* -------------------------------------------------------------------------- */

func (obj HmmPosterior) Dim() int {
	return obj.Hmm.Dim()
}

func (obj HmmPosterior) Eval(r Vector, x ConstVector) error {
	if r.Dim() != x.Dim() {
		return fmt.Errorf("r has invalid length")
	}
	if p, err := obj.PosteriorMarginals(x); err != nil {
		return err
	} else {
		t := NewBareReal(0.0)
		for i := 0; i < x.Dim(); i++ {
			r.At(i).SetValue(math.Inf(-1))
			for j := 0; j < len(obj.States); j++ {
				r.At(i).LogAdd(r.At(i), p[obj.States[j]].At(i), t)
			}
			r.At(i).Exp(r.At(i))
		}
	}
	return nil
}
