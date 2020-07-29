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
import "math/rand"
import "os"
import "testing"

/* -------------------------------------------------------------------------- */

func TestSparseRealVector1(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  w := NewSparseReal64Vector([]int{1,    210,310,30,10192}, []float64{1,  3, 4,-5, 6}, 20000)
  r := NewSparseReal64Vector([]int{1,    210,310,30,10192}, []float64{1,  9,16,25,36}, 20000)
  t := NullReal64()

  v.VmulV(v, w)

  if t.Vnorm(r.VsubV(r, v)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector1Const(test *testing.T) {

  v := NewSparseReal64Vector      ([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  w := NewSparseConstFloat64Vector([]int{1,    210,310,30,10192}, []float64{1,  3, 4,-5, 6}, 20000)
  r := NewSparseReal64Vector      ([]int{1,    210,310,30,10192}, []float64{1,  9,16,25,36}, 20000)
  t := NullReal64()

  v.VmulV(v, w)

  if t.Vnorm(r.VsubV(r, v)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector2(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3,4,-5,6}, 20000)
  w := NewSparseReal64Vector(nil, nil, 0)
  t := NullReal64()

  if v_bytes, err := json.Marshal(v); err != nil {
    test.Errorf("test failed")
  } else {
    if err := json.Unmarshal(v_bytes, w); err != nil {
      test.Errorf("test failed")
    } else {
      if t.Vnorm(v.VsubV(v, w)); t.GetFloat64() > 0.0 {
        test.Errorf("test failed")
      }
    }
  }
}

func TestSparseRealVector3(test *testing.T) {

  v  := NewSparseReal64Vector([]int{    1,  100,    210,   310,    30, 10192}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  w1 := NewSparseReal64Vector([]int{19999,     1, 19996, 19997,     0, 19998}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  w2 := NewSparseReal64Vector([]int{    0, 19998,     3,     2, 19999,     1}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  t  := NullReal64()

  v.Sort(false)
  if t.Vnorm(w1.VsubV(w1, v)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }

  v.Sort(true)
  if t.Vnorm(w2.VsubV(w2, v)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector4(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192      }, []float64{1,2,3,4,-5,6   }, 20000)
  w := v.AppendScalar(NewReal64(30))
  r := NewSparseReal64Vector([]int{1,100,210,310,30,10192,20000}, []float64{1,2,3,4,-5,6,30}, 20001)
  t := NullReal64()

  if t.Vnorm(w.VsubV(w, r)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector4Const(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192      }, []float64{1,2,3,4,-5,6   }, 20000)
  w := v.AppendScalar(NewReal64(30))
  r := NewSparseConstFloat64Vector([]int{1,100,210,310,30,10192,20000}, []float64{1,2,3,4,-5,6,30}, 20001)
  t := NullReal64()

  if t.Vnorm(w.VsubV(w, r)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector5(test *testing.T) {

  i := 101
  j := 400

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3,4,-5,6}, 20000)
  w := v.Slice(i,j)
  r := NewSparseReal64Vector([]int{210-i,310-i}, []float64{3,4}, j-i)
  t := NullReal64()

  if t.Vnorm(r.VsubV(r, w)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector5Const(test *testing.T) {

  i := 101
  j := 400

  v := NewSparseConstFloat64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3,4,-5,6}, 20000)
  w := v.ConstSlice(i,j)
  r := NewSparseReal64Vector([]int{210-i,310-i}, []float64{3,4}, j-i)
  t := NullReal64()

  if t.Vnorm(r.VsubV(r, w)); t.GetFloat64() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealVector6(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  r := []float64{1.0, -5.0, 2.0, 3.0, 4.0, 3.785, 6.0}

  i := 0
  for it := v.ITERATOR(); it.Ok(); it.Next() {
    // insert new value
    if i == 2 {
      v.AT(1000).SetFloat64(3.785)
    }
    if s := it.GET(); s.GetFloat64() != r[i] {
      test.Error("test failed")
    }
    i++
  }
  if i != len(r) {
    test.Error("test failed")
  }
}

func TestSparseRealVector7(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  r := []float64{1.0, -5.0, 2.0, 3.0, 6.0}

  i := 0
  for it := v.ITERATOR(); it.Ok(); it.Next() {
    // insert new value
    if i == 2 {
      v.AT(310).SetFloat64(0.0)
      // apply changes
      for is := v.ITERATOR(); is.Ok(); is.Next() {}
    }
    if s := it.GET(); i >= len(r) || s.GetFloat64() != r[i] {
      test.Error("test failed")
    }
    i++
  }
  if i != len(r) {
    test.Error("test failed")
  }
}

func TestSparseRealVector8(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  r := []float64{1.0, -5.0, 2.0, 110.0, 3.0, 4.0, 6.0}

  i := 0
  for it := v.ITERATOR(); it.Ok(); it.Next() {
    if s := it.GET(); i >= len(r) || s.GetFloat64() != r[i] {
      test.Error("test failed")
    }
    // insert new value
    if i == 2 {
      v.AT(100).SetFloat64(0.0)
      // apply changes
      for is := v.ITERATOR(); is.Ok(); is.Next() {}
      v.AT(110).SetFloat64(110.0)
    }
    i++
  }
  if i != len(r) {
    test.Error("test failed")
  }
}

func TestSparseRealVector9(test *testing.T) {

  v := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  r := NewSparseReal64Vector([]int{1,100,210,310,30,10192}, []float64{1,3,2, 4,-5, 6}, 20000)
  t := NullReal64()

  v.Swap(100, 210)
  if t.Vnorm(r.VsubV(r, v)); t.GetFloat64() > 0 {
    test.Errorf("test failed")
  }
}

func TestSparseRealImportExportVector(t *testing.T) {

  filename := "vector_sparse_real_test.table"

  n := 50000
  v := NullSparseReal64Vector(n)
  w := &SparseReal64Vector{}

  // fill vector with values
  for i := 0; i < n; i++ {
    v.At(i).SetFloat64(float64(i))
  }
  if err := v.Export(filename); err != nil {
    panic(err)
  }
  if err := w.Import(filename); err != nil {
    panic(err)
  }
  s := NullFloat64()

  if w.Dim() != n {
    t.Error("test failed")
  } else {
    if s.Vnorm(v.VsubV(v, w)).GetFloat64() != 0.0 {
      t.Error("test failed")
    }
  }
  os.Remove(filename)
}

/* -------------------------------------------------------------------------- */

func TestSparseRealVector10(t *testing.T) {
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

    d1 := AsDenseReal64Vector(v1)
    d2 := AsDenseReal64Vector(v2)

    v2.VaddV(v1, v2)
    d2.VaddV(d1, d2)

    if !d2.Equals(v2, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    v2 = NewSparseReal64Vector(inds2, vals2, n)
    v1.VaddV(v1, v2)

    if !d2.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    d1 = AsDenseReal64Vector(v1)
    s := NewReal64(r.Float64())
    v1.VaddS(v1, s)
    d1.VaddS(d1, s)

    if !d1.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealVector11(t *testing.T) {
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

    d1 := AsDenseReal64Vector(v1)
    d2 := AsDenseReal64Vector(v2)

    v2.VsubV(v1, v2)
    d2.VsubV(d1, d2)

    if !d2.Equals(v2, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    v2 = NewSparseReal64Vector(inds2, vals2, n)
    v1.VsubV(v1, v2)

    if !d2.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    d1 = AsDenseReal64Vector(v1)
    s := NewReal64(r.Float64())
    v1.VsubS(v1, s)
    d1.VsubS(d1, s)

    if !d1.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealVector12(t *testing.T) {
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

    d1 := AsDenseReal64Vector(v1)
    d2 := AsDenseReal64Vector(v2)

    v2.VmulV(v1, v2)
    d2.VmulV(d1, d2)

    if !d2.Equals(v2, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    v2 = NewSparseReal64Vector(inds2, vals2, n)
    v1.VmulV(v1, v2)

    if !d2.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    d1 = AsDenseReal64Vector(v1)
    s := NewReal64(r.Float64())
    v1.VmulS(v1, s)
    d1.VmulS(d1, s)

    if !d1.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealVector13(t *testing.T) {
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

    d1 := AsDenseReal64Vector(v1)
    d2 := AsDenseReal64Vector(v2)

    v2.VdivV(v1, v2)
    d2.VdivV(d1, d2)

    if !d2.Equals(v2, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    v2 = NewSparseReal64Vector(inds2, vals2, n)
    v1.VdivV(v1, v2)

    if !d2.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }

    v1 = NewSparseReal64Vector(inds1, vals1, n)
    d1 = AsDenseReal64Vector(v1)
    s := NewReal64(r.Float64())
    v1.VdivS(v1, s)
    d1.VdivS(d1, s)

    if !d1.Equals(v1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealVector14(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    inds1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    v := NewSparseReal64Vector(inds1, vals1, n)
    w := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    r := NewSparseReal64Vector([]int{}, []float64{}, n)

    d_v := AsDenseReal64Vector(v)
    d_w := AsDenseReal64Matrix(w)
    d_r := AsDenseReal64Vector(r)

      r.MdotV(  w,   v)
    d_r.MdotV(d_w, d_v)

    if !d_r.Equals(r, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseRealVector15(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    inds1 := randn(r, n, m)
    vals1 := randf(r, m)
    rows2 := randn(r, n, m)
    cols2 := randn(r, n, m)
    vals2 := randf(r, m)

    v := NewSparseReal64Vector(inds1, vals1, n)
    w := NewSparseReal64Matrix(rows2, cols2, vals2, n, n)
    r := NewSparseReal64Vector([]int{}, []float64{}, n)

    d_v := AsDenseReal64Vector(v)
    d_w := AsDenseReal64Matrix(w)
    d_r := AsDenseReal64Vector(r)

      r.VdotM(  v,   w)
    d_r.VdotM(d_v, d_w)

    if !d_r.Equals(r, 1e-8) {
      t.Errorf("test failed")
    }
  }
}
