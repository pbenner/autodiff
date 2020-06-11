/* Copyright (C) 2016-2017 Philipp Benner
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

package scalarClassifier

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type LikelihoodClassifier struct {
	FgDist ScalarPdf
	BgDist ScalarPdf
	r1, r2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewLikelihoodClassifier(fgDist ScalarPdf, bgDist ScalarPdf) (*LikelihoodClassifier, error) {
	// determine scalar type
	t := fgDist.ScalarType()
	return &LikelihoodClassifier{fgDist.CloneScalarPdf(), bgDist.CloneScalarPdf(), NewScalar(t, 0.0), NewScalar(t, 0.0)}, nil
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Clone() *LikelihoodClassifier {
	r, _ := NewLikelihoodClassifier(c.FgDist, c.BgDist)
	return r
}

func (c *LikelihoodClassifier) CloneScalarBatchClassifier() ScalarBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Eval(r Scalar, x ConstScalar) error {
	if c.BgDist == nil {
		return c.FgDist.LogPdf(r, x)
	}
	if err := c.FgDist.LogPdf(c.r1, x); err != nil {
		return err
	}
	if err := c.BgDist.LogPdf(c.r2, x); err != nil {
		return err
	}
	r.Sub(c.r1, c.r2)
	return nil
}

/* -------------------------------------------------------------------------- */

type SymmetricClassifier struct {
	LikelihoodClassifier
	v Scalar
	t Scalar
	z Scalar
}

/* -------------------------------------------------------------------------- */

func NewSymmetricClassifier(fgDist ScalarPdf, bgDist ScalarPdf) (*SymmetricClassifier, error) {
	if classifier, err := NewLikelihoodClassifier(fgDist, bgDist); err != nil {
		return nil, err
	} else {
		v := NewScalar(classifier.r1.Type(), 0.0)
		t := NewScalar(classifier.r1.Type(), 0.0)
		z := NewScalar(classifier.r1.Type(), math.Log(0.5))
		return &SymmetricClassifier{*classifier, v, t, z}, nil
	}
}
