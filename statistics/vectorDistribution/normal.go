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
import "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"

import "github.com/pbenner/autodiff/algorithm/determinant"
import "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type NormalDistribution struct {
	Mu       Vector
	Sigma    Matrix
	SigmaInv Matrix
	SigmaDet Scalar
	logH     Scalar
	cl       ConstScalar
	// state
	t1 Vector
	t2 Vector
	t3 Vector
}

/* -------------------------------------------------------------------------- */

func NewNormalDistribution(mu Vector, sigma Matrix) (*NormalDistribution, error) {

	n, m := sigma.Dims()
	t := mu.ElementType()

	if n != m {
		return nil, fmt.Errorf("NewNormalDistribution(): sigma is not a square matrix!")
	}
	if n != mu.Dim() {
		return nil, fmt.Errorf("NewNormalDistribution(): dimensions of mu and sigma do not match!")
	}
	sigmaInv, err := matrixInverse.Run(sigma, matrixInverse.PositiveDefinite{true})
	if err != nil {
		return nil, err
	}
	sigmaDet, err := determinant.Run(sigma, determinant.PositiveDefinite{true})
	if err != nil {
		return nil, err
	}

	if sigmaDet.GetValue() == 0.0 {
		return nil, fmt.Errorf("NewNormalDistribution(): covariance matrix is singular")
	}

	t1 := NewScalar(t, 0.0)

	// -1/2 [ p log(2pi) + log|Sigma| ]
	c := ConstReal(float64(n) * math.Log(2*math.Pi))
	h := NewScalar(t, 0.0)
	h.Mul(ConstReal(-0.5), t1.Add(c, t1.Log(t1.Abs(sigmaDet))))

	result := NormalDistribution{
		Mu:       mu,
		Sigma:    sigma,
		SigmaInv: sigmaInv,
		SigmaDet: sigmaDet,
		logH:     h,
		cl:       ConstReal(math.Log(2.0)),
		t1:       NullVector(t, n),
		t2:       NullVector(t, n),
		t3:       NullVector(t, 1)}

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) Clone() *NormalDistribution {
	return &NormalDistribution{
		Mu:       dist.Mu.CloneVector(),
		Sigma:    dist.Sigma.CloneMatrix(),
		SigmaInv: dist.SigmaInv.CloneMatrix(),
		SigmaDet: dist.SigmaDet.CloneScalar(),
		logH:     dist.logH.CloneScalar(),
		cl:       dist.cl,
		t1:       dist.t1.CloneVector(),
		t2:       dist.t2.CloneVector(),
		t3:       dist.t3.CloneVector()}
}

func (obj *NormalDistribution) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) Dim() int {
	return dist.Mu.Dim()
}

func (dist *NormalDistribution) ScalarType() ScalarType {
	return dist.Mu.ElementType()
}

func (dist *NormalDistribution) Mean() Vector {
	return dist.Mu.CloneVector()
}

func (dist *NormalDistribution) Variance() Vector {
	return dist.Sigma.Diag()
}

func (dist *NormalDistribution) LogH(x ConstVector) Scalar {
	return dist.logH
}

func (dist *NormalDistribution) LogPdf(r Scalar, x ConstVector) error {
	if x.Dim() != dist.Dim() {
		return fmt.Errorf("input vector has invalid dimension")
	}
	// -1/2 [ p log(2pi) + log|Sigma| ]
	h := dist.LogH(x)
	// -1/2 (x-mu)^T Sigma^-1 (x-mu)
	y := dist.t1
	s := dist.t2
	y.VsubV(x, dist.Mu)
	s.VdotM(y, dist.SigmaInv)
	r.VdotV(s, y)
	r.Div(r, ConstReal(2.0))
	r.Sub(h, r)

	return nil
}

func (dist *NormalDistribution) Pdf(r Scalar, x ConstVector) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

func (dist *NormalDistribution) EllipticCdf(r Scalar, x ConstVector) error {
	d, _ := scalarDistribution.NewChiSquaredDistribution(dist.ScalarType(), 2)
	y := dist.t1
	s := dist.t2
	t := dist.t3.At(0)
	// t = T(x) = (x-mu)^T Sigma^-1 (x-mu)
	y.VsubV(x, dist.Mu)
	s.VdotM(y, dist.SigmaInv)
	t.VdotV(s, y)
	// T(x) ~ chi^2_2
	return d.Cdf(r, t)
}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) GetParameters() Vector {
	p := dist.Mu
	p = p.AppendVector(dist.Sigma.AsVector())
	return p
}

func (dist *NormalDistribution) SetParameters(parameters Vector) error {
	n := dist.Dim()
	mu := parameters.Slice(0, n)
	sigma := parameters.Slice(n, n+n*n).AsMatrix(n, n)
	if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	n, ok := config.GetNamedParameterAsInt("N")
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	mu, ok := config.GetNamedParametersAsVector("Mu", t)
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	sigma, ok := config.GetNamedParametersAsMatrix("Sigma", t, n, n)
	if !ok {
		return fmt.Errorf("invalid config file")
	}

	if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *NormalDistribution) ExportConfig() ConfigDistribution {

	n := obj.Dim()

	config := struct {
		Mu    []float64
		Sigma []float64
		N     int
	}{}
	config.Mu = obj.Mu.GetValues()
	config.Sigma = obj.Sigma.GetValues()
	config.N = n

	return NewConfigDistribution("vector:normal distribtion", config)
}
