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
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type LikelihoodClassifier struct {
  FgDist ScalarPdf
  BgDist ScalarPdf
  r1, r2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewLikelihoodClassifier(fgDist ScalarPdf, bgDist ScalarPdf, args... interface{}) (*LikelihoodClassifier, error) {
  // determine scalar type
  t := BareRealType
  for _, arg := range args {
    switch v := arg.(type) {
    case ScalarType:
      t = v
    }
  }
  return &LikelihoodClassifier{fgDist.CloneScalarPdf(), bgDist.CloneScalarPdf(), NewScalar(t, 0.0), NewScalar(t, 0.0)}, nil
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Clone() *LikelihoodClassifier {
  r, _ := NewLikelihoodClassifier(c.FgDist, c.BgDist, c.r1.Type())
  return r
}

func (c *LikelihoodClassifier) CloneScalarBatchClassifier() ScalarBatchClassifier {
  return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Eval(r Scalar, x Scalar) error {
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

func NewSymmetricClassifier(fgDist ScalarPdf, bgDist ScalarPdf, args... interface{}) (*SymmetricClassifier, error) {
  if classifier, err := NewLikelihoodClassifier(fgDist, bgDist, args...); err != nil {
    return nil, err
  } else {
    v := NewScalar(classifier.r1.Type(), 0.0)
    t := NewScalar(classifier.r1.Type(), 0.0)
    z := NewScalar(classifier.r1.Type(), math.Log(0.5))
    return &SymmetricClassifier{*classifier, v, t, z}, nil
  }
}

/* -------------------------------------------------------------------------- */

type PosteriorClassifier struct {
  LikelihoodClassifier
  LogWeights [2]Scalar
}

/* -------------------------------------------------------------------------- */

func NewPosteriorClassifier(fgDist ScalarPdf, bgDist ScalarPdf, weights [2]float64, args... interface{}) (*PosteriorClassifier, error) {
  if classifier, err := NewLikelihoodClassifier(fgDist, bgDist, args...); err != nil {
    return nil, err
  } else {
    r := PosteriorClassifier{}
    r.LikelihoodClassifier = *classifier
    r.LogWeights[0] = NewBareReal(math.Log(weights[0]/(weights[0] + weights[1])))
    r.LogWeights[1] = NewBareReal(math.Log(weights[1]/(weights[0] + weights[1])))
    return &r, nil
  }
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorClassifier) Clone() *PosteriorClassifier {
  logWeights   := [2]Scalar{}
  logWeights[0] = c.LogWeights[0].CloneScalar()
  logWeights[1] = c.LogWeights[1].CloneScalar()
  return &PosteriorClassifier{*c.LikelihoodClassifier.Clone(), logWeights}
}

func (c *PosteriorClassifier) CloneScalarBatchClassifier() ScalarBatchClassifier {
  return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c PosteriorClassifier) Eval(r Scalar, x Scalar) error {
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
  r .Sub(r1, r2)
  return nil
}

/* -------------------------------------------------------------------------- */

type PosteriorOddsClassifier struct {
  LikelihoodClassifier
  LogWeights [2]Scalar
}

/* -------------------------------------------------------------------------- */

func NewPosteriorOddsClassifier(fgDist ScalarPdf, bgDist ScalarPdf, weights [2]float64, args... interface{}) (*PosteriorOddsClassifier, error) {
  if classifier, err := NewLikelihoodClassifier(fgDist, bgDist); err != nil {
    return nil, err
  } else {
    r := PosteriorOddsClassifier{}
    r.LikelihoodClassifier = *classifier
    r.LogWeights[0] = NewBareReal(math.Log(weights[0]/(weights[0] + weights[1])))
    r.LogWeights[1] = NewBareReal(math.Log(weights[1]/(weights[0] + weights[1])))
    return &r, nil
  }
}

/* -------------------------------------------------------------------------- */

func (c *PosteriorOddsClassifier) Clone() *PosteriorOddsClassifier {
  logWeights   := [2]Scalar{}
  logWeights[0] = c.LogWeights[0].CloneScalar()
  logWeights[1] = c.LogWeights[1].CloneScalar()
  return &PosteriorOddsClassifier{*c.LikelihoodClassifier.Clone(), logWeights}
}

func (c *PosteriorOddsClassifier) CloneScalarBatchClassifier() ScalarBatchClassifier {
  return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c PosteriorOddsClassifier) Eval(r Scalar, x Scalar) error {
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
