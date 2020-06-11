/* Copyright (C) 2015 Philipp Benner
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

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "encoding/json"
import "math"
import "io/ioutil"
import "os"
import "testing"

/* -------------------------------------------------------------------------- */

func TestMatrix1(t *testing.T) {
	m1 := NewMatrix(RealType, 4, 7, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28})
	m2 := m1.Slice(1, 3, 2, 5)
	r := NewMatrix(RealType, 2, 3, []float64{10, 11, 12, 17, 18, 19})

	if !m2.Equals(r, 1e-12) {
		t.Error("test failed")
	}
}

func TestMatrix2(t *testing.T) {
	m1 := NewMatrix(RealType, 4, 7, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28})
	m1 = m1.T()
	m2 := m1.Slice(2, 5, 1, 3)
	r := NewMatrix(RealType, 3, 2, []float64{10, 17, 11, 18, 12, 19})

	if !m2.Equals(r, 1e-12) {
		t.Error("test failed")
	}
}

func TestMatrix(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	m2 := m1.T()

	if m1.At(1, 2).GetValue() != m2.At(2, 1).GetValue() {
		t.Error("Matrix transpose failed!")
	}
}

func TestMatrixRowCol(t *testing.T) {

	var m Matrix

	m = NewMatrix(RealType, 3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9})

	r1 := m.Row(1)
	m = m.T()
	r2 := m.Col(1)

	m = m.T()

	s1 := m.Col(1)
	m = m.T()
	s2 := m.Row(1)

	r := NullReal()

	if r.Vnorm(r1.VsubV(r1, r2)).GetValue() > 1e-8 {
		t.Error("Matrix Row/Col() test failed!")
	}
	if r.Vnorm(s1.VsubV(s1, s2)).GetValue() > 1e-8 {
		t.Error("Matrix Row/Col() test failed!")
	}
}

func TestMatrixDiag(t *testing.T) {

	m := NewMatrix(RealType, 3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	v := m.Diag()

	if v.At(0).GetValue() != 1 ||
		v.At(1).GetValue() != 5 ||
		v.At(2).GetValue() != 9 {
		t.Error("Matrix diag failed!")
	}
}

func TestMatrixReference(t *testing.T) {

	m := NewMatrix(RealType, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	c := NewReal(163)

	m.At(0, 0).Set(c)
	c.SetValue(400)

	if m.At(0, 0).GetValue() != 163 {
		t.Error("Matrix transpose failed!")
	}

	r := NullReal()

	m.At(1, 2).Sub(m.At(1, 2), r.Mul(m.At(1, 1), m.At(1, 2)))

	if m.At(1, 2).GetValue() != -24 {
		t.Error("Matrix transpose failed!")
	}
}

func TestMatrixTrace(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	s := NullReal()

	if s.Mtrace(m1).GetValue() != 5 {
		t.Error("Wrong matrix trace!")
	}
}

func TestMatrixDot(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	m2 := m1.T()
	m3 := NullMatrix(RealType, 2, 2)
	m3.MdotM(m1, m2)

	if m3.At(0, 0).GetValue() != 14 {
		t.Error("Matrix multiplication failed!")
	}
}

func TestMatrixMul(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	m2 := NewMatrix(RealType, 2, 3, []float64{6, 5, 4, 3, 2, 1})
	m3 := m1.CloneMatrix()
	m3.MmulM(m1, m2)

	if m3.At(0, 0).GetValue() != 6 {
		t.Error("Matrix multiplication failed!")
	}
}

func TestMatrixMdotV(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	v1 := NewVector(RealType, []float64{1, 2})
	v2 := v1.CloneVector()
	v2.MdotV(m1, v1)
	v3 := NewVector(RealType, []float64{5, 11})

	r := NullReal()

	if r.Vnorm(v2.VsubV(v2, v3)).GetValue() > 1e-8 {
		t.Error("Matrix/Vector multiplication failed!")
	}
}

func TestMatrixVdotM(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	v1 := NewVector(RealType, []float64{1, 2})
	v2 := v1.CloneVector()
	v2.VdotM(v1, m1)
	v3 := NewVector(RealType, []float64{7, 10})

	r := NullReal()

	if r.Vnorm(v2.VsubV(v2, v3)).GetValue() > 1e-8 {
		t.Error("Matrix/Vector multiplication failed!")
	}
}

func TestMatrixMapReduce(t *testing.T) {

	r1 := NewMatrix(RealType, 2, 2, []float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01})
	r2 := 84.79103
	t1 := NewReal(0.0)
	m := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
	m.Map(func(x Scalar) { x.Exp(x) })
	a := m.Reduce(func(x Scalar, y ConstScalar) Scalar { return x.Add(x, y) }, t1)
	s := NullReal()

	if s.Mnorm(m.MsubM(m, r1)).GetValue() > 1e-8 {
		t.Error("Matrix/Vector multiplication failed!")
	}
	if math.Abs(a.GetValue()-r2) > 1e-2 {
		t.Error("Vector map/reduce failed!")
	}
}

func TestOuter(t *testing.T) {
	a := NewVector(RealType, []float64{1, 3, 2})
	b := NewVector(RealType, []float64{2, 1, 0, 3})
	r := NullMatrix(RealType, 3, 4)
	r.Outer(a, b)
	m := NewMatrix(RealType, 3, 4, []float64{
		2, 1, 0, 3,
		6, 3, 0, 9,
		4, 2, 0, 6})
	s := NullReal()

	if s.Mnorm(r.MsubM(r, m)).GetValue() > 1e-8 {
		t.Error("Outer product multiplication failed!")
	}

}

func TestMatrixJacobian(t *testing.T) {

	r1 := NullReal()
	r2 := NullReal()

	f := func(x ConstVector) ConstVector {
		if x.Dim() != 2 {
			panic("Invalid input vector!")
		}
		y := NullVector(RealType, 3)
		// x1^2 + y^2 - 6
		y.At(0).Sub(r1.Add(r1.Pow(x.ConstAt(0), NewBareReal(2)), r2.Pow(x.ConstAt(1), NewBareReal(2))), NewBareReal(6))
		// x^3 - y^2
		y.At(1).Sub(r1.Pow(x.ConstAt(0), NewBareReal(3)), r2.Pow(x.ConstAt(1), NewBareReal(2)))
		y.At(2).SetValue(2)

		return y
	}

	v1 := NewVector(RealType, []float64{1, 1})
	m1 := NullMatrix(RealType, 3, 2)
	m2 := NewMatrix(RealType, 3, 2, []float64{2, 2, 3, -2, 0, 0})
	s := NullReal()

	m1.Jacobian(f, v1)

	if s.Mnorm(m1.MsubM(m1, m2)).GetValue() > 1e-8 {
		t.Error("Jacobian test failed!")
	}
}

func TestMatrixHessian(t *testing.T) {
	x := NewVector(RealType, []float64{1.5, 2.5})
	k := NewReal(3.0)

	Variables(2, x.At(0), x.At(1))

	t1 := NullReal()
	t2 := NullReal()

	// y = x^3 + y^3 + 3xy
	f := func(x ConstVector) ConstScalar {
		return t1.Sub(t1.Add(t1.Pow(x.ConstAt(0), k), t2.Pow(x.ConstAt(1), k)), t2.Mul(NewReal(3.0), t2.Mul(x.ConstAt(0), x.ConstAt(1))))
	}
	r1 := NullMatrix(RealType, 2, 2)
	r2 := NewMatrix(RealType, 2, 2, []float64{
		9, -3,
		-3, 15})
	s := NullReal()

	r1.Hessian(f, x)

	if s.Mnorm(r1.MsubM(r1, r2)).GetValue() > 1e-8 {
		t.Error("Matrix Hessian test failed!")
	}
}

func TestReadMatrix(t *testing.T) {

	filename := "matrix_dense_test.table"

	m := &DenseRealMatrix{}

	if err := m.Import(filename); err != nil {
		panic(err)
	}
	r := NewMatrix(RealType, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	s := NullReal()

	if s.Mnorm(m.MsubM(m, r)).GetValue() != 0.0 {
		t.Error("Read matrix failed!")
	}
}

func TestSymmetricPermutation(t *testing.T) {

	m1 := NewMatrix(RealType, 4, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16})
	m2 := m1.CloneMatrix()

	// define permutation
	pi := make([]int, 4)
	pi[0] = 0
	pi[1] = 3
	pi[2] = 2
	pi[3] = 1

	m2.SymmetricPermutation(pi)
	m2.SymmetricPermutation(pi)

	s := NullReal()

	if s.Mnorm(m1.MsubM(m1, m2)).GetValue() > 1e-20 {
		t.Error("SymmetricPermutation() failed")
	}
}

func TestMdotM(t *testing.T) {
	r1 := NewMatrix(RealType, 3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9})
	r2 := NewMatrix(RealType, 3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9})
	r3 := NullMatrix(RealType, 3, 3)
	r3.MdotM(r1, r2)
	s := NullMatrix(RealType, 3, 3)
	q := NullReal()

	{
		r1 := r1.CloneMatrix()
		r2 := r1.CloneMatrix()
		r1.MdotM(r1, r2)

		if q.Mnorm(s.MsubM(r1, r3)).GetValue() > 1e-8 {
			t.Error("test failed")
		}
	}
	{
		r1 := r1.CloneMatrix()
		r2 := r1.CloneMatrix()
		r2.MdotM(r1, r2)

		if q.Mnorm(s.MsubM(r2, r3)).GetValue() > 1e-8 {
			t.Error("test failed")
		}
	}
}

func TestMatrixJson(t *testing.T) {

	writeJson := func(filename string, obj interface{}) error {
		if f, err := os.Create(filename); err != nil {
			return err
		} else {
			b, err := json.MarshalIndent(obj, "", "  ")
			if err != nil {
				return err
			}
			if _, err := f.Write(b); err != nil {
				return err
			}
		}
		return nil
	}
	readJson := func(filename string, obj interface{}) error {
		if f, err := os.Open(filename); err != nil {
			return err
		} else {
			buffer, err := ioutil.ReadAll(f)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(buffer, obj); err != nil {
				return err
			}
		}
		return nil
	}
	{
		filename := "matrix_dense_test.1.json"

		r1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
		r2 := &DenseRealMatrix{}

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, r2); err != nil {
			t.Error(err)
			return
		}
		if r1.At(0, 0).GetValue() != r2.At(0, 0).GetValue() {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
	{
		filename := "matrix_dense_test.2.json"

		r1 := NewMatrix(RealType, 2, 2, []float64{1, 2, 3, 4})
		r1.At(0, 0).Alloc(1, 2)
		r1.At(0, 0).SetDerivative(0, 2.3)
		r2 := &DenseRealMatrix{}

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, r2); err != nil {
			t.Error(err)
			return
		}
		if r1.At(0, 0).GetValue() != r2.At(0, 0).GetValue() {
			t.Error("test failed")
		}
		if r1.At(0, 0).GetDerivative(0) != r2.At(0, 0).GetDerivative(0) {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
}

func TestMatrixTransposeInPlace(t *testing.T) {

	m1 := NewMatrix(RealType, 2, 4, []float64{11, 12, 13, 14, 21, 22, 23, 24})
	m1.Tip()
	m2 := NewMatrix(RealType, 4, 2, []float64{11, 21, 12, 22, 13, 23, 14, 24})

	s := NullReal()

	if s.Mnorm(m1.MsubM(m1, m2)).GetValue() != 0.0 {
		t.Error("test failed")
	}
}
