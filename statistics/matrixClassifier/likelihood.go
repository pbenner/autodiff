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

package matrixClassifier

/* -------------------------------------------------------------------------- */

import "fmt"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type LikelihoodClassifier struct {
	FgDist MatrixPdf
	BgDist MatrixPdf
	r1, r2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewLikelihoodClassifier(fgDist MatrixPdf, bgDist MatrixPdf) (*LikelihoodClassifier, error) {
	// determine scalar type
	t := fgDist.ScalarType()
	if bgDist != nil {
		n1, n2 := fgDist.Dims()
		m1, m2 := bgDist.Dims()
		if n1 != m1 || n2 != m2 {
			return nil, fmt.Errorf("foreground and background dimensions do not match (foreground has dimension `%dx%d' whereas the background has dimension `%dx%d')", n1, n2, m1, m2)
		}
		bgDist = bgDist.CloneMatrixPdf()
	}
	fgDist = fgDist.CloneMatrixPdf()
	return &LikelihoodClassifier{fgDist, bgDist, NewScalar(t, 0.0), NewScalar(t, 0.0)}, nil
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Clone() *LikelihoodClassifier {
	r, err := NewLikelihoodClassifier(c.FgDist, c.BgDist)
	if err != nil {
		panic(err)
	}
	return r
}

func (c *LikelihoodClassifier) CloneMatrixBatchClassifier() MatrixBatchClassifier {
	return c.Clone()
}

/* -------------------------------------------------------------------------- */

func (c *LikelihoodClassifier) Dims() (int, int) {
	return c.FgDist.Dims()
}

func (c *LikelihoodClassifier) Eval(r Scalar, x ConstMatrix) error {
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
