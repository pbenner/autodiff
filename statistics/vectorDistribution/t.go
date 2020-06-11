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
import "github.com/pbenner/autodiff/algorithm/determinant"
import "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type TDistribution struct {
	Nu       Scalar
	Mu       Vector
	Sigma    Matrix
	SigmaInv Matrix
	SigmaDet Scalar
	c1       Scalar
	np       Scalar
	t1       Vector
	t2       Vector
	z        Scalar
}

/* -------------------------------------------------------------------------- */

func NewTDistribution(nu Scalar, mu Vector, sigma Matrix) (*TDistribution, error) {

	t := nu.Type()
	t1 := NewScalar(t, 0.0)

	n, m := sigma.Dims()

	if n != m {
		return nil, fmt.Errorf("NewTDistribution(): sigma is not a square matrix!")
	}
	if n != mu.Dim() {
		return nil, fmt.Errorf("NewTDistribution(): dimensions of mu and sigma do not match!")
	}
	sigmaInv, err := matrixInverse.Run(sigma, matrixInverse.PositiveDefinite{true})
	if err != nil {
		return nil, err
	}
	sigmaDet, err := determinant.Run(sigma, determinant.PositiveDefinite{true})
	if err != nil {
		return nil, err
	}

	d2 := ConstReal(float64(n) / 2.0)
	n2 := NewScalar(t, 0.0)
	n2.Div(nu, ConstReal(2.0))
	np := NewScalar(t, 0.0)
	np.Add(n2, d2)
	// +log Gamma(nu/2 + d/2)

	z := NewScalar(t, 0.0)
	z.Lgamma(np)
	// -log Gamma(nu/2)
	z.Sub(z, t1.Lgamma(n2))
	// -1/2 log |Sigma|
	z.Sub(z, t1.Div(t1.Log(sigmaDet), ConstReal(2.0)))
	// -d/2 log nu*pi
	z.Sub(z, t1.Mul(d2, t1.Log(t1.Mul(nu, ConstReal(math.Pi)))))

	result := TDistribution{
		Nu:       nu.CloneScalar(),
		Mu:       mu.CloneVector(),
		Sigma:    sigma,
		SigmaInv: sigmaInv,
		SigmaDet: sigmaDet,
		np:       np,
		t1:       NullVector(t, n),
		t2:       NullVector(t, n),
		z:        z}

	return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *TDistribution) Clone() *TDistribution {
	return &TDistribution{
		Nu:       dist.Nu.CloneScalar(),
		Mu:       dist.Mu.CloneVector(),
		Sigma:    dist.Sigma.CloneMatrix(),
		SigmaInv: dist.SigmaInv.CloneMatrix(),
		SigmaDet: dist.SigmaDet.CloneScalar(),
		np:       dist.np.CloneScalar(),
		t1:       dist.t1.CloneVector(),
		t2:       dist.t2.CloneVector(),
		z:        dist.z.CloneScalar()}
}

func (obj *TDistribution) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *TDistribution) ScalarType() ScalarType {
	return dist.Nu.Type()
}

func (dist *TDistribution) Dim() int {
	return dist.Mu.Dim()
}

func (dist *TDistribution) Mean() (Vector, error) {
	if dist.Nu.GetValue() <= 1.0 {
		return nil, fmt.Errorf("mean undefined for given parameters")
	}
	return dist.Mu.CloneVector(), nil
}

func (dist *TDistribution) Variance() (Vector, error) {
	if dist.Nu.GetValue() <= 2.0 {
		return nil, fmt.Errorf("variance undefined for given parameters")
	}
	n := dist.Dim()
	t := NullScalar(dist.ScalarType())
	m := NullMatrix(dist.ScalarType(), n, n)
	m.MmulS(dist.Sigma, t.Div(dist.Nu, t.Sub(dist.Nu, ConstReal(2.0))))

	return m.Diag(), nil
}

func (dist *TDistribution) LogPdf(r Scalar, x ConstVector) error {
	y := dist.t1
	s := dist.t2
	// 1 + 1/nu (x-mu)^T Sigma^-1 (x-mu)
	y.VsubV(x, dist.Mu)
	s.VdotM(y, dist.SigmaInv)
	r.VdotV(s, y)
	r.Div(r, dist.Nu)
	r.Add(r, ConstReal(1.0))
	// log r^[(v+p)/2]
	r.Log(r)
	r.Mul(r, dist.np)
	r.Sub(dist.z, r)

	return nil
}

func (dist *TDistribution) Pdf(r Scalar, x ConstVector) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist *TDistribution) GetParameters() Vector {
	p := dist.Mu
	p = p.AppendVector(dist.Sigma.AsVector())
	return p
}

func (dist *TDistribution) SetParameters(parameters Vector) error {
	n := dist.Dim()
	nu := parameters.At(0)
	parameters = parameters.Slice(1, parameters.Dim())
	mu := parameters.Slice(0, n)
	parameters = parameters.Slice(n, parameters.Dim())
	sigma := parameters.Slice(0, n*n).AsMatrix(n, n)
	if tmp, err := NewTDistribution(nu, mu, sigma); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *TDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	n, ok := config.GetNamedParameterAsInt("N")
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	nu, ok := config.GetNamedParameterAsScalar("Nu", t)
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

	if tmp, err := NewTDistribution(nu, mu, sigma); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *TDistribution) ExportConfig() ConfigDistribution {

	n := obj.Dim()

	config := struct {
		Nu    float64
		Mu    []float64
		Sigma []float64
		N     int
	}{}
	config.Nu = obj.Nu.GetValue()
	config.Mu = obj.Mu.GetValues()
	config.Sigma = obj.Sigma.GetValues()
	config.N = n

	return NewConfigDistribution("vector:t distribtion", config)
}
