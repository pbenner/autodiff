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

package vectorClassifier

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type LikelihoodClassifier struct {
	FgDist VectorPdf
	BgDist VectorPdf
	r1, r2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewLikelihoodClassifier(fgDist VectorPdf, bgDist VectorPdf) (*LikelihoodClassifier, error) {
	// determine scalar type
	t := fgDist.ScalarType()
	if bgDist != nil {
		n := fgDist.Dim()
		m := bgDist.Dim()
		if n != m {
			return nil, fmt.Errorf("foreground and background dimensions do not match (foreground has dimension `%d' whereas the background has dimension `%d')", n, m)
		}
	}
	return &LikelihoodClassifier{fgDist.CloneVectorPdf(), bgDist.CloneVectorPdf(), NewScalar(t, 0.0), NewScalar(t, 0.0)}, nil
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Clone() *LikelihoodClassifier {
	r, _ := NewLikelihoodClassifier(c.FgDist, c.BgDist)
	return r
}

func (c *LikelihoodClassifier) CloneVectorBatchClassifier() VectorBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Dim() int {
	return c.FgDist.Dim()
}

func (c *LikelihoodClassifier) Eval(r Scalar, x ConstVector) error {
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

func (classifier *LikelihoodClassifier) Linearize() error {
	var fgDist *vectorDistribution.NormalDistribution
	var bgDist *vectorDistribution.NormalDistribution
	if classifier.FgDist == nil {
		return fmt.Errorf("linearization failed: foreground distribution is nil")
	}
	if classifier.BgDist == nil {
		return fmt.Errorf("linearization failed: background distribution is nil")
	}
	switch dist := classifier.FgDist.(type) {
	case *vectorDistribution.NormalDistribution:
		fgDist = dist
	default:
		return fmt.Errorf("linearization failed: foreground model is not a normal distribution")
	}
	switch dist := classifier.BgDist.(type) {
	case *vectorDistribution.NormalDistribution:
		bgDist = dist
	default:
		return fmt.Errorf("linearization failed: background model is not a normal distribution")
	}
	/* get foreground parameters */
	mu1, sigma1 := fgDist.Mu, fgDist.Sigma

	/* get background parameters */
	mu2, sigma2 := bgDist.Mu, bgDist.Sigma

	// to get a linear classifier, both covariance matrices must
	// be the same
	sigma := sigma1.CloneMatrix()
	sigma.MdivS(sigma.MaddM(sigma1, sigma2), NewBareReal(2.0))

	// copy new parameters back to distribution
	if tmp, err := vectorDistribution.NewNormalDistribution(mu1, sigma.CloneMatrix()); err != nil {
		return err
	} else {
		fgDist = tmp
	}
	if tmp, err := vectorDistribution.NewNormalDistribution(mu2, sigma.CloneMatrix()); err != nil {
		return err
	} else {
		bgDist = tmp
	}
	classifier.FgDist = fgDist
	classifier.BgDist = bgDist
	return nil
}

/* -------------------------------------------------------------------------- */

type SymmetricClassifier struct {
	LikelihoodClassifier
	v Vector
	t Scalar
	z Scalar
}

/* -------------------------------------------------------------------------- */

func NewSymmetricClassifier(fgDist VectorPdf, bgDist VectorPdf) (*SymmetricClassifier, error) {
	if classifier, err := NewLikelihoodClassifier(fgDist, bgDist); err != nil {
		return nil, err
	} else {
		v := NullVector(classifier.r1.Type(), classifier.Dim())
		t := NewScalar(classifier.r1.Type(), 0.0)
		z := NewScalar(classifier.r1.Type(), math.Log(0.5))
		return &SymmetricClassifier{*classifier, v, t, z}, nil
	}
}

/* -------------------------------------------------------------------------- */

func (c *SymmetricClassifier) Clone() *SymmetricClassifier {
	r, _ := NewSymmetricClassifier(c.FgDist, c.BgDist)
	return r
}

func (c *SymmetricClassifier) CloneVectorBatchClassifier() VectorBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *SymmetricClassifier) Eval(r Scalar, x1 ConstVector) error {
	if x1.Dim() != c.Dim() {
		return fmt.Errorf("evaluating classifier failed: input vector has invalid dimension")
	}
	x2 := c.v
	r1 := c.r1
	r2 := c.r2
	// copy data to x2 in reverse order
	for i := 0; i < x1.Dim(); i++ {
		x2.At(i).Set(x1.ConstAt(x1.Dim() - i - 1))
	}
	if c.BgDist == nil {
		if err := c.FgDist.LogPdf(r1, x1); err != nil {
			return err
		}
		if err := c.FgDist.LogPdf(r2, x2); err != nil {
			return err
		}
		r.LogAdd(r1, r2, c.t)
		r.Add(r, c.z)
	} else {
		if err := c.FgDist.LogPdf(r1, x1); err != nil {
			return err
		}
		if err := c.FgDist.LogPdf(r2, x2); err != nil {
			return err
		}
		r.LogAdd(r1, r2, c.t)
		if err := c.BgDist.LogPdf(r2, x1); err != nil {
			return err
		}
		r.Sub(r, r2)
		r.Add(r, c.z)
	}
	return nil
}
