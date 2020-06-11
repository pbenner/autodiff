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

type BetaDistribution struct {
	Alpha    Scalar
	Beta     Scalar
	as1      Scalar
	bs1      Scalar
	z        Scalar
	c1       Scalar
	t1       Scalar
	t2       Scalar
	LogScale bool
}

/* -------------------------------------------------------------------------- */

func NewBetaDistribution(alpha, beta Scalar, logScale bool) (*BetaDistribution, error) {
	if alpha.GetValue() <= 0.0 || beta.GetValue() <= 0.0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	t := alpha.Type()
	dist := BetaDistribution{}
	dist.Alpha = alpha.CloneScalar()
	dist.Beta = beta.CloneScalar()
	dist.as1 = alpha.CloneScalar()
	dist.bs1 = beta.CloneScalar()
	dist.as1.Sub(alpha, NewBareReal(1.0))
	dist.bs1.Sub(beta, NewBareReal(1.0))
	dist.LogScale = logScale

	t1 := alpha.CloneScalar()
	t1.Add(t1, beta)
	t1.Lgamma(t1)
	t2 := alpha.CloneScalar()
	t2.Lgamma(t2)
	t3 := beta.CloneScalar()
	t3.Lgamma(t3)
	t1.Sub(t1, t2)
	t1.Sub(t1, t3)
	dist.z = t1
	if logScale {
		dist.c1 = NewScalar(t, 0.0)
	} else {
		dist.c1 = NewScalar(t, 1.0)
	}
	dist.t1 = NewScalar(t, 0.0)
	dist.t2 = NewScalar(t, 0.0)
	return &dist, nil
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) Clone() *BetaDistribution {
	r, _ := NewBetaDistribution(dist.Alpha, dist.Beta, dist.LogScale)
	return r
}

func (dist *BetaDistribution) CloneScalarPdf() ScalarPdf {
	return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) ScalarType() ScalarType {
	return dist.Alpha.Type()
}

func (dist *BetaDistribution) LogPdf(r Scalar, x ConstScalar) error {
	if dist.LogScale {
		if v := x.GetValue(); v > 0.0 {
			r.SetValue(math.Inf(-1))
			return nil
		}
	} else {
		if v := x.GetValue(); v < 0.0 || v > 1.0 {
			r.SetValue(math.Inf(-1))
			return nil
		}
	}
	t1 := dist.t1
	t2 := dist.t2

	if dist.LogScale {
		if v := dist.bs1.GetValue(); v == 0.0 {
			t2.SetValue(0.0)
		} else {
			// t2 = log(1-theta)
			t2.LogSub(dist.c1, x, t1)
			// t2 = beta*log(1-theta)
			t2.Mul(t2, dist.bs1)
		}
		if v := dist.as1.GetValue(); v == 0.0 {
			t1.SetValue(0.0)
		} else {
			// t1 = alpha*log(theta)
			t1.Mul(dist.as1, x)
		}
	} else {
		if v := dist.bs1.GetValue(); v == 0.0 {
			t2.SetValue(0.0)
		} else {
			// t2 = 1-theta
			t2.Sub(dist.c1, x)
			// t2 = log(1-theta)
			t2.Log(t2)
			// t2 = beta*log(1-theta)
			t2.Mul(t2, dist.bs1)
		}
		if v := dist.as1.GetValue(); v == 0.0 {
			t1.SetValue(0.0)
		} else {
			// t1 = log(theta)
			t1.Log(x)
			// t1 = alpha*log(theta)
			t1.Mul(t1, dist.as1)
		}
	}

	r.Add(t1, t2)
	r.Add(r, dist.z)

	if math.IsNaN(r.GetValue()) {
		return fmt.Errorf("NaN value detected for input value `%v' and parameters `%v'", x, dist.GetParameters())
	}
	return nil
}

func (dist *BetaDistribution) Pdf(r Scalar, x ConstScalar) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) GetParameters() Vector {
	p := NullVector(dist.ScalarType(), 3)
	p.At(0).Set(dist.Alpha)
	p.At(1).Set(dist.Beta)
	if dist.LogScale {
		p.At(2).SetValue(1.0)
	} else {
		p.At(2).SetValue(0.0)
	}
	return p
}

func (dist *BetaDistribution) SetParameters(parameters Vector) error {
	if tmp, err := NewBetaDistribution(parameters.At(0), parameters.At(1), parameters.At(2).GetValue() == 1.0); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	if parameters, ok := config.GetParametersAsFloats(); !ok {
		return fmt.Errorf("invalid config file")
	} else {
		alpha := NewScalar(t, parameters[0])
		beta := NewScalar(t, parameters[1])
		logScale := parameters[2] == 1.0

		if tmp, err := NewBetaDistribution(alpha, beta, logScale); err != nil {
			return err
		} else {
			*dist = *tmp
		}
		return nil
	}
}

func (dist *BetaDistribution) ExportConfig() (config ConfigDistribution) {

	return NewConfigDistribution("scalar:beta distribution", dist.GetParameters())
}
