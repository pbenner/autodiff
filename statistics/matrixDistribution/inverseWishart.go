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

package matrixDistribution

/* -------------------------------------------------------------------------- */

import "fmt"

//import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

import "github.com/pbenner/autodiff/algorithm/determinant"
import "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type InverseWishartDistribution struct {
	Nu   Scalar
	S    Matrix
	SDet Scalar
	d    Scalar
	z    Scalar
	c1   Scalar
	c2   Scalar
	// state
	t         Scalar
	inSituDet determinant.InSitu
	inSituInv matrixInverse.InSitu
}

/* -------------------------------------------------------------------------- */

func NewInverseWishartDistribution(nu Scalar, s Matrix) (*InverseWishartDistribution, error) {

	t := nu.Type()
	t1 := NewScalar(t, 0.0)
	t2 := NewScalar(t, 0.0)

	n, m := s.Dims()

	if n != m {
		return nil, fmt.Errorf("NewInverseWishartDistribution(): S is not a square matrix!")
	}
	sDet, err := determinant.Run(s, determinant.PositiveDefinite{true})
	if err != nil {
		return nil, err
	}
	d := NewScalar(t, float64(n))
	c1 := NewBareReal(1.0)
	c2 := NewBareReal(2.0)
	// negative log partition function
	z := NewScalar(t, 0.0)
	z.Mul(t1.Div(nu, c2), t2.Log(sDet))                     // |S|^(nu/2)
	z.Sub(z, t1.Mul(t1.Mul(nu, t1.Div(d, c2)), t2.Log(c2))) // 2^(nu n/2)
	z.Sub(z, t1.Mlgamma(t1.Div(nu, c2), n))                 // Gamma_n(nu/2)

	result := InverseWishartDistribution{
		Nu:   nu,
		S:    s,
		SDet: sDet,
		d:    d,
		c1:   c1,
		c2:   c2,
		z:    z,
		t:    NewScalar(t, 0.0)}

	return &result, nil

}

/* -------------------------------------------------------------------------- */

func (obj *InverseWishartDistribution) Clone() *InverseWishartDistribution {
	return &InverseWishartDistribution{
		Nu:   obj.Nu.CloneScalar(),
		S:    obj.S.CloneMatrix(),
		SDet: obj.SDet.CloneScalar(),
		d:    obj.d.CloneScalar(),
		c1:   obj.c1.CloneScalar(),
		c2:   obj.c2.CloneScalar(),
		z:    obj.z.CloneScalar(),
		t:    obj.t.CloneScalar()}
}

func (obj *InverseWishartDistribution) CloneMatrixPdf() MatrixPdf {
	return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *InverseWishartDistribution) ScalarType() ScalarType {
	return obj.Nu.Type()
}

func (obj *InverseWishartDistribution) dim() int {
	n, _ := obj.S.Dims()
	return n
}

func (obj *InverseWishartDistribution) Dims() (int, int) {
	return obj.S.Dims()
}

func (obj *InverseWishartDistribution) Mean() (Matrix, error) {
	n := obj.dim()
	if obj.Nu.GetValue() <= float64(n)-1.0 {
		return nil, fmt.Errorf("mean is not defined for the given parameters")
	}
	t1 := NullScalar(obj.ScalarType())
	t2 := NullMatrix(obj.ScalarType(), n, n)
	return t2.MdivS(obj.S, t1.Sub(t1.Sub(obj.Nu, obj.d), ConstReal(1.0))), nil
}

func (obj *InverseWishartDistribution) Variance() (Matrix, error) {
	n := obj.dim()
	if obj.Nu.GetValue() <= float64(n)-1.0 {
		return nil, fmt.Errorf("variance is not defined for the given parameters")
	}
	m := NullMatrix(RealType, n, n)
	// some constants
	t1 := NewScalar(obj.ScalarType(), 0.0)
	t2 := NewScalar(obj.ScalarType(), 0.0)
	c1 := NewScalar(obj.ScalarType(), 0.0)
	c2 := NewScalar(obj.ScalarType(), 0.0)
	c3 := NewScalar(obj.ScalarType(), 0.0)
	c4 := NewScalar(obj.ScalarType(), 0.0)
	c5 := NewScalar(obj.ScalarType(), 0.0)
	c1.Sub(obj.Nu, obj.d)                  // (nu - d)
	c2.Add(c1, ConstReal(1.0))             // (nu - d + 1)
	c3.Sub(c1, ConstReal(1.0))             // (nu - d - 1)
	c4.Sub(c1, ConstReal(3.0))             // (nu - d - 3)
	c5.Mul(t1.Mul(c1, t1.Mul(c3, c3)), c4) // (nu - d)(nu - d - 1)^2 (nu - d - 3)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := t1
			y := t2
			x.Mul(obj.S.At(i, j), obj.S.At(i, j))
			y.Mul(obj.S.At(i, i), obj.S.At(j, j))
			m.At(i, j).Div(x.Add(x.Mul(c2, x), y.Mul(c3, y)), c5)
		}
	}
	return m, nil
}

func (obj *InverseWishartDistribution) LogPdf(r Scalar, x ConstMatrix) error {
	xInv, err1 := matrixInverse.Run(x, matrixInverse.PositiveDefinite{true}, &obj.inSituInv)
	xDet, err2 := determinant.Run(x, determinant.PositiveDefinite{true}, &obj.inSituDet)
	if err1 != nil {
		return err1
	}
	if err2 != nil {
		return err2
	}
	xDet.Log(xDet)
	xInv.MmulM(obj.S, xInv)
	t := obj.t
	t.Mtrace(xInv)
	t.Div(t, obj.c2)
	// density
	r.Add(obj.Nu, obj.d)
	r.Add(r, obj.c1)
	r.Div(r, obj.c2)
	r.Mul(r, xDet)
	r.Neg(r)
	r.Sub(r, t)
	r.Add(r, obj.z)
	return nil
}

func (obj *InverseWishartDistribution) Pdf(r Scalar, x ConstMatrix) error {
	if err := obj.LogPdf(r, x); err != nil {
		return err
	}
	r.Exp(r)
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *InverseWishartDistribution) GetParameters() Vector {
	p := obj.S.AsVector()
	p = p.AppendScalar(obj.Nu)
	return p
}

func (obj *InverseWishartDistribution) SetParameters(parameters Vector) error {
	n := obj.dim()
	s := parameters.Slice(0, n*n).AsMatrix(n, n)
	nu := parameters.At(n * n)
	if tmp, err := NewInverseWishartDistribution(nu, s); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func (obj *InverseWishartDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

	n, ok := config.GetNamedParameterAsInt("N")
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	nu, ok := config.GetNamedParameterAsScalar("Nu", t)
	if !ok {
		return fmt.Errorf("invalid config file")
	}
	sigma, ok := config.GetNamedParametersAsMatrix("Sigma", t, n, n)
	if !ok {
		return fmt.Errorf("invalid config file")
	}

	if tmp, err := NewInverseWishartDistribution(nu, sigma); err != nil {
		return err
	} else {
		*obj = *tmp
	}
	return nil
}

func (obj *InverseWishartDistribution) ExportConfig() ConfigDistribution {

	n := obj.dim()

	config := struct {
		Nu    float64
		Sigma []float64
		N     int
	}{}
	config.Nu = obj.Nu.GetValue()
	config.Sigma = obj.S.GetValues()
	config.N = n

	return NewConfigDistribution("matrix:inverse wishart distribtion", config)
}
