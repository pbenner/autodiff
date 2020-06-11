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

package scalarDistribution

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type BinomialDistribution struct {
	Theta Scalar
	n     Scalar
	np1   Scalar
	z     Scalar
	c1    Scalar
	ct    Scalar
	t1    Scalar
	t2    Scalar
}

/* -------------------------------------------------------------------------- */

func NewBinomialDistribution(theta Scalar, n int) (*BinomialDistribution, error) {
	if theta.GetValue() < 0.0 || theta.GetValue() > 1.0 || n < 0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	t := theta.Type()
	dist := BinomialDistribution{}

	dist.Theta = NewScalar(t, 0.0)
	dist.Theta.Log(theta)
	dist.n = NewScalar(t, float64(n+0))
	dist.np1 = NewScalar(t, float64(n+1))

	// z = Gamma(n+1)
	dist.z = NewScalar(t, float64(n+1))
	dist.z.Lgamma(dist.z)
	// c1 = 1
	dist.c1 = NewScalar(t, 1.0)
	// ct = Log(1-theta)
	dist.ct = NewScalar(t, 1.0)
	dist.ct.Sub(dist.ct, theta)
	dist.ct.Log(dist.ct)
	// temporary memory
	dist.t1 = NewScalar(t, 0.0)
	dist.t2 = NewScalar(t, 0.0)

	return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) Clone() *BinomialDistribution {
	t := NewScalar(dist.ScalarType(), 0.0)
	t.Exp(dist.Theta)
	r, _ := NewBinomialDistribution(t, int(dist.n.GetValue()))
	return r
}

func (dist *BinomialDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) ScalarType() ScalarType {
	return dist.Theta.Type()
}

func (dist *BinomialDistribution) GetN() int {
	return int(dist.n.GetValue())
}

func (dist *BinomialDistribution) SetN(n int) error {
	if n < 0 {
		return fmt.Errorf("invalid parameter")
	}
	dist.n.SetValue(float64(n + 0))
	dist.np1.SetValue(float64(n + 1))
	dist.z.Lgamma(dist.np1)
	return nil
}

func (dist *BinomialDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if v := x.GetValue(); v < 0.0 || math.Floor(v) != v {
		r.SetValue(math.Inf(-1))
		return nil
	}
	t1 := dist.t1
	t2 := dist.t2

	// Gamma(k+1)
	t1.Add(x, dist.c1)
	t1.Lgamma(t1)

	// Gamma(n-k+1)
	t2.Sub(dist.np1, x)
	t2.Lgamma(t2)

	// Gamma(n+1) / [Gamma(k+1) Gamma(n-k+1)]
	r.Sub(dist.z, t1)
	r.Sub(r, t2)

	// p^k
	t1.Mul(dist.Theta, x)

	// (1-p)^(n-k)
	t2.Sub(dist.n, x)
	t2.Mul(dist.ct, t2)

	// sum up results
	r.Add(r, t1)
	r.Add(r, t2)

	return nil
}

func (dist *BinomialDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 2)
	p.At(0).Set(dist.Theta)
	p.At(1).Set(dist.n)
	return p
}

func (dist *BinomialDistribution) SetParameters(parameters Vector) error {
	t := parameters.At(0)
	t.Exp(t)
	n := parameters.At(1)
	if tmp, err := NewBinomialDistribution(t, int(n.GetValue())); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		theta := NewScalar(t, parameters[0])
		n := int(parameters[1])

		if tmp, err := NewBinomialDistribution(theta, n); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *BinomialDistribution) ExportConfig() ConfigDistribution {

	return NewConfigDistribution("scalar:binomial distribution", dist.GetParameters())
}
