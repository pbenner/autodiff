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

func TestMatrix1(t *testing.T) {
  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28}, 4, 7)
  m2 := m1.Slice(1,3,2,5)
  r  := NewDenseFloat64Matrix([]float64{10, 11, 12, 17, 18, 19}, 2, 3)

  if !m2.Equals(r, 1e-12) {
    t.Error("test failed")
  }
}

func TestMatrix2(t *testing.T) {
  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28}, 4, 7)
  m1  = m1.T().(*DenseFloat64Matrix)
  m2 := m1.Slice(2,5,1,3)
  r  := NewDenseFloat64Matrix([]float64{10, 17, 11, 18, 12, 19}, 3, 2)

  if !m2.Equals(r, 1e-12) {
    t.Error("test failed")
  }
}

func TestMatrix(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6}, 2, 3)
  m2 := m1.T()

  if m1.At(1,2).GetFloat64() != m2.At(2,1).GetFloat64() {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixRowCol(t *testing.T) {

  var m Matrix

  m = NewDenseFloat64Matrix([]float64{
     1,  2,  3,
     4,  5,  6,
     7,  8,  9,
    10, 11, 12}, 4, 3)

  r1 := m.Row(1)
  c1 := m.ConstRow(1)
  m   = m.T()
  r2 := m.Col(1)
  c2 := m.ConstCol(1)

  m   = m.T()

  s1 := m.Col(1)
  d1 := m.ConstCol(1)
  m   = m.T()
  s2 := m.Row(1)
  d2 := m.ConstRow(1)

  r := NullFloat64()

  if r.Vnorm(r1.VsubV(r1, r2)).GetFloat64() > 1e-8 {
    t.Error("Matrix Row/Col() test failed!")
  }
  if r.Vnorm(r1.VsubV(c1, c2)).GetFloat64() > 1e-8 {
    t.Error("Matrix Row/Col() test failed!")
  }
  if r.Vnorm(s1.VsubV(s1, s2)).GetFloat64() > 1e-8 {
    t.Error("Matrix Row/Col() test failed!")
  }
  if r.Vnorm(s2.VsubV(d1, d2)).GetFloat64() > 1e-8 {
    t.Error("Matrix Row/Col() test failed!")
  }

  m = NewDenseFloat64Matrix([]float64{
     1,  2,  3,
     4,  5,  6,
     7,  8,  9,
    10, 11, 12}, 4, 3)

  r1 = m.Row(1)
  r1.At(0).SetFloat64(100.0)
  r2 = m.Col(1)
  r2.At(0).SetFloat64(100.0)

  if m.At(1, 0).GetFloat64() != 4.0 {
    t.Error("Matrix Row/Col() test failed!")
  }
  if m.At(0, 1).GetFloat64() != 2.0 {
    t.Error("Matrix Row/Col() test failed!")
  }
}

func TestMatrixDiag(t *testing.T) {

  m := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6,7,8,9}, 3, 3)
  v := m.Diag()

  if v.At(0).GetFloat64() != 1 ||
     v.At(1).GetFloat64() != 5 ||
     v.At(2).GetFloat64() != 9 {
    t.Error("Matrix diag failed!")
  }
}

func TestMatrixReference(t *testing.T) {

  m := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6}, 2, 3)
  c := NewFloat64(163)

  m.At(0, 0).Set(c)
  c.SetFloat64(400)

  if m.At(0,0).GetFloat64() != 163 {
    t.Error("Matrix transpose failed!")
  }

  r := NullFloat64()

  m.At(1, 2).Sub(m.At(1,2), r.Mul(m.At(1,1), m.At(1,2)))

  if m.At(1,2).GetFloat64() != -24 {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixTrace(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  s := NullFloat64()

  if s.Mtrace(m1).GetFloat64() != 5 {
    t.Error("Wrong matrix trace!")
  }
}

func TestMatrixDot(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6}, 2, 3)
  m2 := m1.T()
  m3 := NullDenseFloat64Matrix(2, 2)
  m3.MdotM(m1, m2)

  if m3.At(0,0).GetFloat64() != 14 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixMul(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6}, 2, 3)
  m2 := NewDenseFloat64Matrix([]float64{6,5,4,3,2,1}, 2, 3)
  m3 := m1.CloneMatrix()
  m3.MmulM(m1, m2)

  if m3.At(0,0).GetFloat64() != 6 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixMdotV(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  v1 := NewDenseFloat64Vector([]float64{1, 2})
  v2 := v1.CloneVector()
  v2.MdotV(m1, v1)
  v3 := NewDenseFloat64Vector([]float64{5, 11})

  r := NullFloat64()

  if r.Vnorm(v2.VsubV(v2, v3)).GetFloat64() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixVdotM(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  v1 := NewDenseFloat64Vector([]float64{1, 2})
  v2 := v1.CloneVector()
  v2.VdotM(v1, m1)
  v3 := NewDenseFloat64Vector([]float64{7, 10})

  r := NullFloat64()

  if r.Vnorm(v2.VsubV(v2, v3)).GetFloat64() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixMapReduce(t *testing.T) {

  r1 := NewDenseFloat64Matrix([]float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01}, 2, 2)
  r2 := 84.79103
  t1 := NewFloat64(0.0)
  m := NewDenseFloat64Matrix([]float64{1, 2,3,4}, 2, 2)
  m.Map(func(x Scalar) { x.Exp(x) })
  a := m.Reduce(func(x Scalar, y ConstScalar) Scalar { x.Add(x, y); return x }, t1)
  s := NullFloat64()

  if s.Mnorm(m.MsubM(m, r1)).GetFloat64() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
  if math.Abs(a.GetFloat64() - r2) > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
}

func TestOuter(t *testing.T) {
  a := NewDenseFloat64Vector([]float64{1,3,2})
  b := NewDenseFloat64Vector([]float64{2,1,0,3})
  r := NullDenseFloat64Matrix(3, 4)
  r.Outer(a,b)
  m := NewDenseFloat64Matrix([]float64{
    2,1,0,3,
    6,3,0,9,
    4,2,0,6 }, 3, 4)
  s := NullFloat64()

  if s.Mnorm(r.MsubM(r, m)).GetFloat64() > 1e-8  {
    t.Error("Outer product multiplication failed!")
  }

}

func TestReadMatrix(t *testing.T) {

  filename := "matrix_dense_test.table"

  m := &DenseFloat64Matrix{}
  
  if err := m.Import(filename); err != nil {
    panic(err)
  }
  r := NewDenseFloat64Matrix([]float64{1,2,3,4,5,6}, 2, 3)
  s := NullFloat64()

  if s.Mnorm(m.MsubM(m, r)).GetFloat64() != 0.0 {
    t.Error("Read matrix failed!")
  }
}

func TestSymmetricPermutation(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16 }, 4, 4)
  m2 := m1.CloneMatrix()

  // define permutation
  pi := make([]int, 4)
  pi[0] = 0
  pi[1] = 3
  pi[2] = 2
  pi[3] = 1

  m2.SymmetricPermutation(pi)
  m2.SymmetricPermutation(pi)

  s := NullFloat64()

  if s.Mnorm(m1.MsubM(m1, m2)).GetFloat64() > 1e-20 {
    t.Error("SymmetricPermutation() failed")
  }
}

func TestMdotM(t *testing.T) {
  r1 := NewDenseFloat64Matrix([]float64{
    1, 2, 3,
    4, 5, 6,
    7, 8, 9 }, 3, 3)
  r2 := NewDenseFloat64Matrix([]float64{
    1, 2, 3,
    4, 5, 6,
    7, 8, 9 }, 3, 3)
  r3 := NullDenseFloat64Matrix(3, 3)
  r3.MdotM(r1, r2)
  s  := NullDenseFloat64Matrix(3, 3)
  q  := NullFloat64()

  {
    r1 := r1.CloneMatrix()
    r2 := r1.CloneMatrix()
    r1.MdotM(r1, r2)

    if q.Mnorm(s.MsubM(r1, r3)).GetFloat64() > 1e-8 {
      t.Error("test failed")
    }
  }
  {
    r1 := r1.CloneMatrix()
    r2 := r1.CloneMatrix()
    r2.MdotM(r1, r2)

    if q.Mnorm(s.MsubM(r2, r3)).GetFloat64() > 1e-8 {
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

    r1 := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
    r2 := &DenseFloat64Matrix{}

    if err := writeJson(filename, r1); err != nil {
      t.Error(err); return
    }
    if err := readJson(filename, r2); err != nil {
      t.Error(err); return
    }
    if r1.At(0,0).GetFloat64() != r2.At(0,0).GetFloat64() {
      t.Error("test failed")
    }
    os.Remove(filename)
  }
}

func TestMatrixTransposeInPlace(t *testing.T) {

  m1 := NewDenseFloat64Matrix([]float64{11,12,13,14,21,22,23,24}, 2, 4)
  m1.Tip()
  m2 := NewDenseFloat64Matrix([]float64{11,21,12,22,13,23,14,24}, 4, 2)

  s  := NullFloat64()

  if s.Mnorm(m1.MsubM(m1, m2)).GetFloat64() != 0.0 {
    t.Error("test failed")
  }
}
