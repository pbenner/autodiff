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

func TestVector(t *testing.T) {

	v := NewVector(RealType, []float64{1, 2, 3, 4, 5, 6})

	if v.At(1).GetValue() != 2.0 {
		t.Error("Vector initialization failed!")
	}
}

func TestVectorSort(t *testing.T) {

	v1 := NewVector(RealType, []float64{4, 3, 7, 4, 1, 29, 6})
	v2 := NewVector(RealType, []float64{4, 3, 7, 4, 1, 29, 6})

	v1.Sort(false)
	v2.Sort(true)

	if v1.At(6).GetValue() != 29.0 {
		t.Error("Vector sorting failed!")
	}
	if v2.At(6).GetValue() != 1.0 {
		t.Error("Vector sorting failed!")
	}
}

func TestVectorAsMatrix(t *testing.T) {

	v := NewVector(RealType, []float64{1, 2, 3, 4, 5, 6})
	m := v.AsMatrix(2, 3)

	if m.At(1, 0).GetValue() != 4 {
		t.Error("Vector to matrix conversion failed!")
	}
}

func TestVdotV(t *testing.T) {

	a := NewVector(RealType, []float64{1, 2, 3, 4})
	b := NewVector(RealType, []float64{2, -1, 1, 7})
	r := NullReal()
	r.VdotV(a, b)

	if r.GetValue() != 31 {
		t.Error("VmulV() failed!")
	}
}

func TestVmulV(t *testing.T) {

	a := NewVector(RealType, []float64{1, 2, 3, 4})
	b := NewVector(RealType, []float64{2, -1, 1, 7})
	r := a.CloneVector()
	r.VmulV(a, b)

	if r.At(1).GetValue() != -2 {
		t.Error("VmulV() failed!")
	}
}

func TestReadVector(t *testing.T) {

	filename := "vector_dense_test.table"

	v := DenseRealVector{}

	if err := v.Import(filename); err != nil {
		panic(err)
	}
	r := NewVector(RealType, []float64{1, 2, 3, 4, 5, 6})
	s := NullReal()

	if s.Vnorm(v.VsubV(v, r)).GetValue() != 0.0 {
		t.Error("Read vector failed!")
	}
}

func TestVectorMapReduce(t *testing.T) {

	r1 := NewVector(RealType, []float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01})
	r2 := 84.79103
	t1 := NewReal(0.0)
	a := NewVector(RealType, []float64{1, 2, 3, 4})
	a.Map(func(x Scalar) { x.Exp(x) })
	b := a.Reduce(func(x Scalar, y ConstScalar) Scalar { return x.Add(x, y) }, t1)
	s := NullReal()

	if s.Vnorm(a.VsubV(a, r1)).GetValue() > 1e-2 {
		t.Error("Vector map/reduce failed!")
	}
	if math.Abs(b.GetValue()-r2) > 1e-2 {
		t.Error("Vector map/reduce failed!")
	}
}

func TestVectorJson(t *testing.T) {

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
		filename := "vector_dense_test.1.json"

		r1 := NewVector(RealType, []float64{1, 2, 3, 4})
		r2 := DenseRealVector{}

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, &r2); err != nil {
			t.Error(err)
			return
		}
		if r1.At(0).GetValue() != r2.At(0).GetValue() {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
	{
		filename := "vector_dense_test.2.json"

		r1 := NewVector(RealType, []float64{1, 2, 3, 4})
		r1.At(0).Alloc(1, 2)
		r1.At(0).SetDerivative(0, 2.3)
		r2 := DenseRealVector{}

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, &r2); err != nil {
			t.Error(err)
			return
		}
		if r1.At(0).GetValue() != r2.At(0).GetValue() {
			t.Error("test failed")
		}
		if r1.At(0).GetDerivative(0) != r2.At(0).GetDerivative(0) {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
}
