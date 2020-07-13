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

    m1 := NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 := NewSparseRealMatrix(n, n, rows2, cols2, vals2)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MaddM(m1, m2)
    d2.MaddM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 = NewSparseRealMatrix(n, n, rows2, cols2, vals2)
    m1.MaddM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
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

    m1 := NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 := NewSparseRealMatrix(n, n, rows2, cols2, vals2)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MsubM(m1, m2)
    d2.MsubM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 = NewSparseRealMatrix(n, n, rows2, cols2, vals2)
    m1.MsubM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
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

    m1 := NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 := NewSparseRealMatrix(n, n, rows2, cols2, vals2)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MmulM(m1, m2)
    d2.MmulM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 = NewSparseRealMatrix(n, n, rows2, cols2, vals2)
    m1.MmulM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
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

    m1 := NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 := NewSparseRealMatrix(n, n, rows2, cols2, vals2)

    d1 := AsDenseRealMatrix(m1)
    d2 := AsDenseRealMatrix(m2)

    m2.MdivM(m1, m2)
    d2.MdivM(d1, d2)

    if !d2.Equals(m2, 1e-8) {
      t.Errorf("test failed")
    }

    m1 = NewSparseRealMatrix(n, n, rows1, cols1, vals1)
    m2 = NewSparseRealMatrix(n, n, rows2, cols2, vals2)
    m1.MdivM(m1, m2)

    if !d2.Equals(m1, 1e-8) {
      t.Errorf("test failed")
    }
  }
}
