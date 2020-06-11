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

package vectorClassifier

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type PosteriorClassifier struct {
	LikelihoodClassifier
	LogWeights [2]ConstScalar
}

/* -------------------------------------------------------------------------- */

func NewPosteriorClassifier(fgDist VectorPdf, bgDist VectorPdf, weights [2]float64) (*PosteriorClassifier, error) {
	if classifier, err := NewLikelihoodClassifier(fgDist, bgDist); err != nil {
		return nil, err
	} else {
		r := PosteriorClassifier{}
		r.LikelihoodClassifier = *classifier
		r.LogWeights[0] = ConstReal(math.Log(weights[0] / (weights[0] + weights[1])))
		r.LogWeights[1] = ConstReal(math.Log(weights[1] / (weights[0] + weights[1])))
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorClassifier) Clone() *PosteriorClassifier {
	logWeights := [2]ConstScalar{}
	logWeights[0] = c.LogWeights[0]
	logWeights[1] = c.LogWeights[1]
	return &PosteriorClassifier{*c.LikelihoodClassifier.Clone(), logWeights}
}

func (c *PosteriorClassifier) CloneVectorBatchClassifier() VectorBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorClassifier) Eval(r Scalar, x ConstVector) error {
	r1 := c.r1
	r2 := c.r2
	if err := c.FgDist.LogPdf(r1, x); err != nil {
		return err
	}
	if err := c.BgDist.LogPdf(r2, x); err != nil {
		return err
	}
	r1.Add(r1, c.LogWeights[0])
	r2.Add(r2, c.LogWeights[1])
	r2.LogAdd(r1, r2, r)
	r.Sub(r1, r2)
	return nil
}

/* -------------------------------------------------------------------------- */

type PosteriorOddsClassifier struct {
	LikelihoodClassifier
	LogWeights [2]ConstScalar
}

/* -------------------------------------------------------------------------- */

func NewPosteriorOddsClassifier(fgDist VectorPdf, bgDist VectorPdf, weights [2]float64) (*PosteriorOddsClassifier, error) {
	if classifier, err := NewLikelihoodClassifier(fgDist, bgDist); err != nil {
		return nil, err
	} else {
		r := PosteriorOddsClassifier{}
		r.LikelihoodClassifier = *classifier
		r.LogWeights[0] = ConstReal(math.Log(weights[0] / (weights[0] + weights[1])))
		r.LogWeights[1] = ConstReal(math.Log(weights[1] / (weights[0] + weights[1])))
		return &r, nil
	}
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorOddsClassifier) Clone() *PosteriorOddsClassifier {
	logWeights := [2]ConstScalar{}
	logWeights[0] = c.LogWeights[0]
	logWeights[1] = c.LogWeights[1]
	return &PosteriorOddsClassifier{*c.LikelihoodClassifier.Clone(), logWeights}
}

func (c *PosteriorOddsClassifier) CloneVectorBatchClassifier() VectorBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorOddsClassifier) Eval(r Scalar, x ConstVector) error {
	r1 := c.r1
	r2 := c.r2
	if err := c.FgDist.LogPdf(r1, x); err != nil {
		return err
	}
	if err := c.BgDist.LogPdf(r2, x); err != nil {
		return err
	}
	r.Sub(r1, r2)
	return nil
}
