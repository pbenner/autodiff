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
import "math/rand"
import "os"
import "testing"

/* -------------------------------------------------------------------------- */

func randn(r *rand.Rand, n, m int) []int {
  a := make([]int, n)
  for i := range a {
    a[i] = i
  }
  r.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
  return a[0:m]
}

func randf(r *rand.Rand, m int) []float64 {
  a := make([]float64, m)
  for i := range a {
    a[i] = r.Float64()
  }
  return a
}

/* -------------------------------------------------------------------------- */

func TestSparseRealMatrixRowCol(t *testing.T) {

  var m Matrix

  m = NewDenseReal64Matrix([]float64{
     1,  2,  3,
     4,  0,  0,
     7,  8,  9,
     0, 11, 12}, 4, 3)
  m = AsSparseReal64Matrix(m)

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

  r := NullReal64()

  if r.Vnorm(r1.VsubV(r1, r2)).GetFloat64() > 1e-8 {
    t.Error("test failed")
  }
  if r.Vnorm(r1.VsubV(c1, c2)).GetFloat64() > 1e-8 {
    t.Error("test failed")
  }
  if r.Vnorm(s1.VsubV(s1, s2)).GetFloat64() > 1e-8 {
    t.Error("test failed")
  }
  if r.Vnorm(s2.VsubV(d1, d2)).GetFloat64() > 1e-8 {
    t.Error("test failed")
  }

  m = NewDenseReal64Matrix([]float64{
     1,  2,  3,
     4,  0,  0,
     7,  8,  9,
     0, 11, 12}, 4, 3)
  m = AsSparseReal64Matrix(m)

  r1 = m.Row(1)
  r1.At(0).SetFloat64(100.0)
  r2 = m.Col(1)
  r2.At(0).SetFloat64(100.0)

  if m.At(1, 0).GetFloat64() != 4.0 {
    t.Error("test failed")
  }
  if m.At(0, 1).GetFloat64() != 2.0 {
    t.Error("test failed")
  }
}

func TestSparseRealImportExportMatrix(t *testing.T) {

  filename := "matrix_sparse_real_test.table"

  n := 10
  v := NullSparseReal64Matrix(2, n/2)
  w := NullSparseReal64Matrix(0, 0)

  // fill vector with values
  for i := 0; i < 2; i++ {
    for j := 0; j < n/2; j++ {
      v.At(i, j).SetFloat64(float64(i*j))
    }
  }
  if err := v.Export(filename); err != nil {
    panic(err)
  }
  if err := w.Import(filename); err != nil {
    panic(err)
  }
  s := NullFloat64()

  if n1, n2 := w.Dims(); n1 != 2 || n2 != n/2 {
    t.Error("test failed")
  } else {
    if s.Mnorm(v.MsubM(v, w)).GetFloat64() != 0.0 {
      t.Error("test failed")
    }
  }
  os.Remove(filename)
}

/* -------------------------------------------------------------------------- */

func TestSparseRealMatrix1(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    rows1 := randn(r, n, m)
    cols1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseReal64Matrix(m1)
    d2 := AsDenseReal64Matrix(m2)

    m2.MaddM(m1, m2)
    d2.MaddM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    m1.MaddM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseReal64Matrix(m1)
    s := NewReal64(r.Float64())
    m1.MaddS(m1, s)
    d1.MaddS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix2(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    rows1 := randn(r, n, m)
    cols1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseReal64Matrix(m1)
    d2 := AsDenseReal64Matrix(m2)

    m2.MsubM(m1, m2)
    d2.MsubM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    m1.MsubM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseReal64Matrix(m1)
    s := NewReal64(r.Float64())
    m1.MsubS(m1, s)
    d1.MsubS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix3(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    rows1 := randn(r, n, m)
    cols1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseReal64Matrix(m1)
    d2 := AsDenseReal64Matrix(m2)

    m2.MmulM(m1, m2)
    d2.MmulM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    m1.MmulM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseReal64Matrix(m1)
    s := NewReal64(r.Float64())
    m1.MmulS(m1, s)
    d1.MmulS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix4(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    rows1 := randn(r, n, m)
    cols1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseReal64Matrix(m1)
    d2 := AsDenseReal64Matrix(m2)

    m2.MdivM(m1, m2)
    d2.MdivM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    m1.MdivM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseReal64Matrix(m1)
    s := NewReal64(r.Float64())
    m1.MdivS(m1, s)
    d1.MdivS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix5(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    inds1 := randn(r, n, m)
    vals1 := randf(r, m)
    inds2 := randn(r, n, m)
    vals2 := randf(r, m)

    v1 := NewSparseReal64Vector(inds1, vals1, n)
    v2 := NewSparseReal64Vector(inds2, vals2, n)
    r  := NewSparseReal64Matrix([]int{}, []int{}, []float64{}, n, n)

    d1 := AsDenseReal64Vector(v1)
    d2 := AsDenseReal64Vector(v2)
    s  := AsDenseReal64Matrix(r)

    r.Outer(v1, v2)
    s.Outer(d1, d2)

    if !r.Equals(s, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix6(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 25
    m := r.Intn(n)

    rows1 := randn(r, n, m)
    cols1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    r  := NewSparseReal64Matrix([]int{}, []int{}, []float64{}, n, n)

    d1 := AsDenseReal64Matrix(m1)
    d2 := AsDenseReal64Matrix(m2)
    s  := AsDenseReal64Matrix(r)

    r.MdotM(m1, m2)
    s.MdotM(d1, d2)

    if !r.Equals(s, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealMatrix7(t *testing.T) {
  r := rand.New(rand.NewSource(3))

  for i := 0; i < 1000; i++ {
    n := 20
    m := r.Intn(n)
    p := r.Intn(n)

    rows1 := randn(r, n+0, m)
    cols1 := randn(r, n+p, m)
    vals1 := randf(r, m)

    m1 := NewSparseReal64Matrix(rows1, cols1, vals1, n+0, n+p)
    m2 := NewSparseReal64Matrix(cols1, rows1, vals1, n+p, n+0).T()

    for it := m1.JointIterator(m2); it.Ok(); it.Next() {
      a, b := it.Get()
      if a.GetFloat64() != b.GetFloat64() {
        t.Errorf("test failed")
      }
    }
  }
}
