/* Copyright (C) 2015-2020 Philipp Benner
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

func TestRealVector(t *testing.T) {

  v := NewDenseReal64Vector([]float64{1,2,3,4,5,6})

  if v.At(1).GetFloat64() != 2.0 {
    t.Error("Vector initialization failed!")
  }
}

func TestRealVectorSort(t *testing.T) {

  v1 := NewDenseReal64Vector([]float64{4,3,7,4,1,29,6})
  v2 := NewDenseReal64Vector([]float64{4,3,7,4,1,29,6})

  v1.Sort(false)
  v2.Sort(true)

  if v1.At(6).GetFloat64() != 29.0 {
    t.Error("Vector sorting failed!")
  }
  if v2.At(6).GetFloat64() != 1.0 {
    t.Error("Vector sorting failed!")
  }
}

func TestRealVectorAsMatrix(t *testing.T) {

  v := NewDenseReal64Vector([]float64{1,2,3,4,5,6})
  m := v.AsMatrix(2, 3)

  if m.At(1,0).GetFloat64() != 4 {
    t.Error("Vector to matrix conversion failed!")
  }
}

func TestRealVdotV(t *testing.T) {

  a := NewDenseReal64Vector([]float64{1, 2,3,4})
  b := NewDenseReal64Vector([]float64{2,-1,1,7})
  r := NullReal64()
  r.VdotV(a, b)

  if r.GetFloat64() != 31 {
    t.Error("VmulV() failed!")
  }
}

func TestRealVmulV(t *testing.T) {

  a := NewDenseReal64Vector([]float64{1, 2,3,4})
  b := NewDenseReal64Vector([]float64{2,-1,1,7})
  r := a.CloneVector()
  r.VmulV(a, b)

  if r.At(1).GetFloat64() != -2 {
    t.Error("VmulV() failed!")
  }
}

func TestRealReadVector(t *testing.T) {

  filename := "vector_dense_test.table"

  v := DenseReal64Vector{}

  if err := v.Import(filename); err != nil {
    panic(err)
  }
  r := NewDenseReal64Vector([]float64{1,2,3,4,5,6})
  s := NullReal64()

  if s.Vnorm(v.VsubV(v, r)).GetFloat64() != 0.0 {
    t.Error("Read vector failed!")
  }
}

func TestRealVectorMapReduce(t *testing.T) {

  r1 := NewDenseReal64Vector([]float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01})
  r2 := 84.79103
  t1 := NewReal64(0.0)
  a := NewDenseReal64Vector([]float64{1, 2,3,4})
  a.Map(func(x Scalar) { x.Exp(x) })
  b := a.Reduce(func(x Scalar, y ConstScalar) Scalar { x.Add(x, y); return x }, t1)
  s := NullReal64()

  if s.Vnorm(a.VsubV(a,r1)).GetFloat64() > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
  if math.Abs(b.GetFloat64() - r2) > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
}

func TestRealVectorJson(t *testing.T) {

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

    r1 := NewDenseReal64Vector([]float64{1,2,3,4})
    r2 := DenseReal64Vector{}

    if err := writeJson(filename, r1); err != nil {
      t.Error(err); return
    }
    if err := readJson(filename, &r2); err != nil {
      t.Error(err); return
    }
    if r1.At(0).GetFloat64() != r2.At(0).GetFloat64() {
      t.Error("test failed")
    }
    os.Remove(filename)
  }
  {
    filename := "vector_dense_test.2.json"

    r1 := NewDenseReal64Vector([]float64{1,2,3,4})
    r1.MagicAt(0).Alloc(1,2)
    r1.MagicAt(0).SetDerivative(0, 2.3)
    r2 := DenseReal64Vector{}

    if err := writeJson(filename, r1); err != nil {
      t.Error(err); return
    }
    if err := readJson(filename, &r2); err != nil {
      t.Error(err); return
    }
    if r1.At(0).GetFloat64() != r2.At(0).GetFloat64() {
      t.Error("test failed")
    }
    if r1.At(0).GetDerivative(0) != r2.At(0).GetDerivative(0) {
      t.Error("test failed")
    }
    os.Remove(filename)
  }
}
