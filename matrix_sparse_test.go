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
import "math/rand"
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

func TestSparseMatrix1(t *testing.T) {
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

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseRealMatrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MaddM(m1, m2)
    d2.MaddM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseRealMatrix(rows2, cols2, vals2, n, n)
    m1.MaddM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseRealMatrix(m1)
    s := NewScalar(RealType, r.Float64())
    m1.MaddS(m1, s)
    d1.MaddS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix2(t *testing.T) {
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

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseRealMatrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MsubM(m1, m2)
    d2.MsubM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseRealMatrix(rows2, cols2, vals2, n, n)
    m1.MsubM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseRealMatrix(m1)
    s := NewScalar(RealType, r.Float64())
    m1.MsubS(m1, s)
    d1.MsubS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix3(t *testing.T) {
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

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseRealMatrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MmulM(m1, m2)
    d2.MmulM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseRealMatrix(rows2, cols2, vals2, n, n)
    m1.MmulM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseRealMatrix(m1)
    s := NewScalar(RealType, r.Float64())
    m1.MmulS(m1, s)
    d1.MmulS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix4(t *testing.T) {
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

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseRealMatrix(rows2, cols2, vals2, n, n)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MdivM(m1, m2)
    d2.MdivM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 = NewSparseRealMatrix(rows2, cols2, vals2, n, n)
    m1.MdivM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    d1 = AsDenseRealMatrix(m1)
    s := NewScalar(RealType, r.Float64())
    m1.MdivS(m1, s)
    d1.MdivS(d1, s)

    if !d1.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix5(t *testing.T) {
  r := rand.New(rand.NewSource(42))

  for i := 0; i < 1000; i++ {
    n := 100
    m := r.Intn(n)

    inds1 := randn(r, n, m)
    vals1 := randf(r, m)
    inds2 := randn(r, n, m)
    vals2 := randf(r, m)

    v1 := NewSparseRealVector(inds1, vals1, n)
    v2 := NewSparseRealVector(inds2, vals2, n)
    r  := NewSparseRealMatrix([]int{}, []int{}, []float64{}, n, n)

    d1 := AsDenseRealVector(v1)
    d2 := AsDenseRealVector(v2)
    s  := AsDenseRealMatrix(r)

    r.Outer(v1, v2)
    s.Outer(d1, d2)

    if !r.Equals(s, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix6(t *testing.T) {
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

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n, n)
    m2 := NewSparseRealMatrix(rows2, cols2, vals2, n, n)
    r  := NewSparseRealMatrix([]int{}, []int{}, []float64{}, n, n)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)
    s  := AsDenseRealMatrix(r)

    r.MdotM(m1, m2)
    s.MdotM(d1, d2)

    if !r.Equals(s, 1e-8) {
      t.Errorf("test failed")
    }
  }
}

func TestSparseMatrix7(t *testing.T) {
  r := rand.New(rand.NewSource(3))

  for i := 0; i < 1000; i++ {
    n := 20
    m := r.Intn(n)
    p := r.Intn(n)

    rows1 := randn(r, n+0, m)
    cols1 := randn(r, n+p, m)
    vals1 := randf(r, m)

    m1 := NewSparseRealMatrix(rows1, cols1, vals1, n+0, n+p)
    m2 := NewSparseRealMatrix(cols1, rows1, vals1, n+p, n+0).T()

    for it := m1.JointIterator(m2); it.Ok(); it.Next() {
      a, b := it.GetValue()
      if a != b {
        t.Errorf("test failed")
      }
    }
  }
}
