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

/* -------------------------------------------------------------------------- */

type SkewNormalDistribution struct {
	Normal1 NormalDistribution
	Normal2 scalarDistribution.NormalDistribution
	Xi      Vector
	Omega   Matrix
	Alpha   Vector
	Scale   Vector
	l2      Scalar
	// state
	r1 Scalar
	r2 Scalar
	t  Scalar
	z  Vector
}

/* -------------------------------------------------------------------------- */

// Azzalini, Adelchi, and Alessandra Dalla Valle. "The multivariate skew-normal
// distribution." Biometrika 83.4 (1996): 715-726.

func NewSkewNormalDistribution(xi Vector, omega Matrix, alpha Vector, scale Vector) (*SkewNormalDistribution, error) {
	t := xi.ElementType()
	t1 := NewScalar(t, 0.0)
	// dimension
	n, m := omega.Dims()
	// check parameter dimensions
	if n != xi.Dim() ||
		(n != alpha.Dim()) ||
		(n != scale.Dim()) ||
		(n != m) {
		return nil, fmt.Errorf("NewSkewNormalDistribution(): Parameter dimensions do not match!")
	}
	// parameters for the multivariate normal
	// kappa = diag(s) omega diag(s)
	kappa := NullMatrix(t, n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			kappa.At(i, j).Mul(t1.Mul(scale.At(i), scale.At(j)), omega.At(i, j))
		}
	}
	// parameters for the standard normal cdf
	mu := NewReal(0.0)
	sigma := NewReal(1.0)

	normal1, err := NewNormalDistribution(xi, kappa)
	if err != nil {
		return nil, err
	}
	normal2, err := scalarDistribution.NewNormalDistribution(mu, sigma)
	if err != nil {
		return nil, err
	}

	result := SkewNormalDistribution{
		Normal1: *normal1,
		Normal2: *normal2,
		Xi:      xi,
		Omega:   omega.CloneMatrix(),
		Alpha:   alpha.CloneVector(),
		Scale:   scale.CloneVector(),
		l2:      NewScalar(t, math.Log(2)),
		r1:      NewScalar(t, 0.0),
		r2:      NewScalar(t, 0.0),
		t:       NewScalar(t, 0.0),
		z:       NullVector(t, n)}

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *SkewNormalDistribution) Clone() *SkewNormalDistribution {
	return &SkewNormalDistribution{
		Normal1: *dist.Normal1.Clone(),
		Normal2: *dist.Normal2.Clone(),
		Xi:      dist.Xi.CloneVector(),
		Omega:   dist.Omega.CloneMatrix(),
		Alpha:   dist.Alpha.CloneVector(),
		Scale:   dist.Scale.CloneVector(),
		l2:      dist.l2.CloneScalar(),
		r1:      dist.r1.CloneScalar(),
		r2:      dist.r2.CloneScalar(),
		t:       dist.t.CloneScalar(),
		z:       dist.z.CloneVector()}
}

func (obj *SkewNormalDistribution) CloneVectorPdf() VectorPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *SkewNormalDistribution) Dim() int {
	return dist.Xi.Dim()
}

func (dist *SkewNormalDistribution) ScalarType() ScalarType {
	return dist.Xi.ElementType()
}

func (dist *SkewNormalDistribution) LogPdf(r0 Scalar, x ConstVector) error {
	n := dist.Normal1.Dim()
	z := dist.z
	t := dist.t
	for i := 0; i < n; i++ {
		t.Sub(x.ConstAt(i), dist.Normal1.Mu.At(i))
		z.At(i).Div(t, dist.Scale.At(i))
	}
	t.VdotV(dist.Alpha, z)

	r1 := dist.r1
	r2 := dist.r2

	if err := dist.Normal1.LogPdf(r1, x); err != nil {
		return err
	}
	if err := dist.Normal2.LogCdf(r2, t); err != nil {
		return err
	}

	r0.Add(r1, r2)
	r0.Add(r0, dist.l2)

	return nil
}

func (dist *SkewNormalDistribution) Pdf(r Scalar, x ConstVector) error {
	if err := dist.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (dist SkewNormalDistribution) GetParameters() Vector {
	p := dist.Xi
	p = p.AppendVector(dist.Omega.AsVector())
	p = p.AppendVector(dist.Alpha)
	p = p.AppendVector(dist.Scale)
	return p
}

func (dist *SkewNormalDistribution) SetParameters(parameters Vector) error {
	n := dist.Dim()
	xi := parameters.Slice(0*n+0*n*n, 1*n+0*n*n)
	omega := parameters.Slice(1*n+0*n*n, 1*n+1*n*n).AsMatrix(n, n)
	alpha := parameters.Slice(1*n+1*n*n, 2*n+1*n*n)
	scale := parameters.Slice(2*n+1*n*n, 3*n+1*n*n)
	if tmp, err := NewSkewNormalDistribution(xi, omega, alpha, scale); err != nil {
		return err
	} else {
		*dist = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *SkewNormalDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	n, ok := config.GetNamedParameterAsInt("N")
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	xi, ok := config.GetNamedParametersAsVector("Xi", t)
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	omega, ok := config.GetNamedParametersAsMatrix("Omega", t, n, n)
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	alpha, ok := config.GetNamedParametersAsVector("Alpha", t)
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	scale, ok := config.GetNamedParametersAsVector("Scale", t)
	if !ok {
		return fmt.Errorf("invalid config file")
	}

	if tmp, err := NewSkewNormalDistribution(xi, omega, alpha, scale); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *SkewNormalDistribution) ExportConfig() ConfigDistribution {

	n := obj.Dim()

	config := struct {
		Xi    []float64
		Omega []float64
		Alpha []float64
		Scale []float64
		N     int
	}{}
	config.Xi = obj.Xi.GetValues()
	config.Omega = obj.Omega.GetValues()
	config.Alpha = obj.Alpha.GetValues()
	config.Scale = obj.Scale.GetValues()
	config.N = n

	return NewConfigDistribution("vector:skew normal distribtion", config)
}
