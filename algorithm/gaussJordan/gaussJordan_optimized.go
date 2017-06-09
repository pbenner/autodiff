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

package gaussJordan

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "errors"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func gaussJordan_RealDense(a, x *DenseMatrix, b DenseVector, submatrix []bool) error {
  t := NewReal(0.0)
  c := NewReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // permutation of the rows
  p := make([]int, n)
  for i := 0; i < n; i++ {
    p[i] = i
  }
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    panic("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    panic("GaussJordan(): b has invalid dimension!")
  }
  // loop over columns
  for i := 0; i < n; i++ {
    if !submatrix[i] {
      continue
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.RealAt(p[j], i).GetValue()) > math.Abs(a.RealAt(p[maxrow], i).GetValue()) {
        maxrow = j
      }
    }
    // swap rows
    p[i], p[maxrow] = p[maxrow], p[i]
    // eliminate column i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      // c = a[j, i] / a[i, i]
      c.RealDiv(a.RealAt(p[j], i), a.RealAt(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.RealMul(a.RealAt(p[i], k), c)
        a.RealAt(p[j], k).RealSub(a.RealAt(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.RealMul(x.RealAt(p[i], k), c)
        x.RealAt(p[j], k).RealSub(x.RealAt(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.RealMul(b[p[i]].(*Real), c)
      b[p[j]].(*Real).RealSub(b[p[j]].(*Real), t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.RealAt(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealAt(p[j], i), b[p[i]].(*Real))
      t.RealDiv(t, c)
      b[p[j]].(*Real).RealSub(b[p[j]].(*Real), t)
      if math.IsNaN(b[p[j]].(*Real).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealAt(p[j], i), x.RealAt(p[i], k))
        t.RealDiv(t, c)
        x.RealAt(p[j], k).RealSub(x.RealAt(p[j], k), t)
        if math.IsNaN(x.RealAt(p[j], k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealAt(p[j], i), a.RealAt(p[i], k))
        t.RealDiv(t, c)
        a.RealAt(p[j], k).RealSub(a.RealAt(p[j], k), t)
        if math.IsNaN(a.RealAt(p[j], k).GetValue()) {
          goto singular
        }
      }
    }
    a.RealAt(p[i], i).RealDiv(a.RealAt(p[i], i), c)
    if math.IsNaN(a.RealAt(p[i], i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealAt(p[i], k).RealDiv(x.RealAt(p[i], k), c)
    }
    // normalize ith element in b
    b[p[i]].(*Real).RealDiv(b[p[i]].(*Real), c)
  }
  if err := a.PermuteRows(p); err != nil {
    return err
  }
  if err := x.PermuteRows(p); err != nil {
    return err
  }
  if err := b.Permute(p); err != nil {
    return err
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}

func gaussJordanUpperTriangular_RealDense(a, x *DenseMatrix, b DenseVector, submatrix []bool) error {
  t := NewReal(0.0)
  c := NewReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    panic("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    panic("GaussJordan(): b has invalid dimension!")
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.RealAt(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealAt(j, i), b.RealAt(i))
      t.RealDiv(t, c)
      b.RealAt(j).RealSub(b.RealAt(j), t)
      if math.IsNaN(b.RealAt(j).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealAt(j, i), x.RealAt(i, k))
        t.RealDiv(t, c)
        x.RealAt(j, k).RealSub(x.RealAt(j, k), t)
        if math.IsNaN(x.RealAt(j, k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealAt(j, i), a.RealAt(i, k))
        t.RealDiv(t, c)
        a.RealAt(j, k).RealSub(a.RealAt(j, k),t)
        if math.IsNaN(a.RealAt(j, k).GetValue()) {
          goto singular
        }
      }
    }
    a.RealAt(i, i).RealDiv(a.RealAt(i, i), c)
    if math.IsNaN(a.RealAt(i, i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealAt(i, k).RealDiv(x.RealAt(i, k), c)
    }
    // normalize ith element in b
    b.RealAt(i).RealDiv(b.RealAt(i), c)
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}

/* -------------------------------------------------------------------------- */

func gaussJordan_BareRealDense(a, x *DenseMatrix, b DenseVector, submatrix []bool) error {
  t := NewBareReal(0.0)
  c := NewBareReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // permutation of the rows
  p := make([]int, n)
  for i := 0; i < n; i++ {
    p[i] = i
  }
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    return errors.New("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    return errors.New("GaussJordan(): b has invalid dimension!")
  }
  // loop over columns
  for i := 0; i < n; i++ {
    if !submatrix[i] {
      continue
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.BareRealAt(p[j], i).GetValue()) > math.Abs(a.BareRealAt(p[maxrow], i).GetValue()) {
        maxrow = j
      }
    }
    // swap rows
    p[i], p[maxrow] = p[maxrow], p[i]
    // eliminate column i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      // c = a[j, i] / a[i, i]
      c.BareRealDiv(a.BareRealAt(p[j], i), a.BareRealAt(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.BareRealMul(a.BareRealAt(p[i], k), c)
        a.BareRealAt(p[j], k).BareRealSub(a.BareRealAt(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.BareRealMul(x.BareRealAt(p[i], k), c)
        x.BareRealAt(p[j], k).BareRealSub(x.BareRealAt(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.BareRealMul(b.BareRealAt(p[i]), c)
      b.BareRealAt(p[j]).BareRealSub(b.BareRealAt(p[j]), t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealAt(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealAt(p[j], i), b.BareRealAt(p[i]))
      t.BareRealDiv(t, c)
      b.BareRealAt(p[j]).BareRealSub(b.BareRealAt(p[j]), t)
      if math.IsNaN(b.BareRealAt(p[j]).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealAt(p[j], i), x.BareRealAt(p[i], k))
        t.BareRealDiv(t, c)
        x.BareRealAt(p[j], k).BareRealSub(x.BareRealAt(p[j], k), t)
        if math.IsNaN(x.BareRealAt(p[j], k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealAt(p[j], i), a.BareRealAt(p[i], k))
        t.BareRealDiv(t, c)
        a.BareRealAt(p[j], k).BareRealSub(a.BareRealAt(p[j], k), t)
        if math.IsNaN(a.BareRealAt(p[j], k).GetValue()) {
          goto singular
        }
      }
    }
    a.BareRealAt(p[i], i).BareRealDiv(a.BareRealAt(p[i], i), c)
    if math.IsNaN(a.BareRealAt(p[i], i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealAt(p[i], k).BareRealDiv(x.BareRealAt(p[i], k), c)
    }
    // normalize ith element in b
    b.BareRealAt(p[i]).BareRealDiv(b.BareRealAt(p[i]), c)
  }
  if err := a.PermuteRows(p); err != nil {
    return err
  }
  if err := x.PermuteRows(p); err != nil {
    return err
  }
  if err := b.Permute(p); err != nil {
    return err
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}

func gaussJordanUpperTriangular_BareRealDense(a, x *DenseMatrix, b DenseVector, submatrix []bool) error {
  t := NewBareReal(0.0)
  c := NewBareReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    return errors.New("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    return errors.New("GaussJordan(): b has invalid dimension!")
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealAt(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealAt(j, i), b.BareRealAt(i))
      t.BareRealDiv(t, c)
      b.BareRealAt(j).BareRealSub(b.BareRealAt(j), t)
      if math.IsNaN(b.BareRealAt(j).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealAt(j, i), x.BareRealAt(i, k))
        t.BareRealDiv(t, c)
        x.BareRealAt(j, k).BareRealSub(x.BareRealAt(j, k), t)
        if math.IsNaN(x.BareRealAt(j, k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealAt(j, i), a.BareRealAt(i, k))
        t.BareRealDiv(t, c)
        a.BareRealAt(j, k).BareRealSub(a.BareRealAt(j, k),t)
        if math.IsNaN(a.BareRealAt(j, k).GetValue()) {
          goto singular
        }
      }
    }
    a.BareRealAt(i, i).BareRealDiv(a.BareRealAt(i, i), c)
    if math.IsNaN(a.BareRealAt(i, i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealAt(i, k).BareRealDiv(x.BareRealAt(i, k), c)
    }
    // normalize ith element in b
    b.BareRealAt(i).BareRealDiv(b.BareRealAt(i), c)
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}
