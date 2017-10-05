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

/* -------------------------------------------------------------------------- */

// True if matrix a equals b.
func (a *DenseMatrix) Equals(b ConstMatrix, epsilon float64) bool {
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("MEqual(): matrix dimensions do not match!")
  }
  for i := 0; i < n1; i++ {
    for j := 0; j < m1; j++ {
      if !a.ConstAt(i, j).Equals(b.ConstAt(i, j), epsilon) {
        return false
      }
    }
  }
  return true
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of two matrices. The result is stored in r.
func (r *DenseMatrix) MaddM(a, b ConstMatrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Add(a.ConstAt(i, j), b.ConstAt(i, j))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Add scalar b to all elements of a. The result is stored in r.
func (r *DenseMatrix) MaddS(a ConstMatrix, b ConstScalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Add(a.ConstAt(i, j), b)
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of two matrices. The result is stored in r.
func (r *DenseMatrix) MsubM(a, b ConstMatrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Sub(a.ConstAt(i, j), b.ConstAt(i, j))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Substract b from all elements of a. The result is stored in r.
func (r *DenseMatrix) MsubS(a ConstMatrix, b ConstScalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Sub(a.ConstAt(i, j), b)
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise multiplication of two matrices. The result is stored in r.
func (r *DenseMatrix) MmulM(a, b ConstMatrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Mul(a.ConstAt(i, j), b.ConstAt(i, j))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Multiply all elements of a with b. The result is stored in r.
func (r *DenseMatrix) MmulS(a ConstMatrix, b ConstScalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Mul(a.ConstAt(i, j), b)
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of two matrices. The result is stored in r.
func (r *DenseMatrix) MdivM(a, b ConstMatrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Div(a.ConstAt(i, j), b.ConstAt(i, j))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Divide all elements of a by b. The result is stored in r.
func (r *DenseMatrix) MdivS(a ConstMatrix, b ConstScalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Div(a.ConstAt(i, j), b)
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Matrix product of a and b. The result is stored in r.
func (r *DenseMatrix) MdotM(a, b ConstMatrix) Matrix {
  n , m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m2 != m || m1 != n2 {
    panic("matrix dimensions do not match!")
  }
  t1 := NullScalar(a.ElementType())
  t2 := NullScalar(a.ElementType())
  if r.storageLocation() == b.storageLocation() {
    t3 := r.tmp1[0:n]
    for j := 0; j < m; j++ {
      for i := 0; i < n; i++ {
        t2.Reset()
        for k := 0; k < m1; k++ {
          t1.Mul(a.ConstAt(i, k), b.ConstAt(k, j))
          t2.Add(t2, t1)
        }
        t3[i].Set(t2)
      }
      for i := 0; i < n; i++ {
        r.At(i, j).Set(t3[i])
      }
    }
  } else {
    t3 := r.tmp2[0:m]
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        t2.Reset()
        for k := 0; k < m1; k++ {
          t1.Mul(a.ConstAt(i, k), b.ConstAt(k, j))
          t2.Add(t2, t1)
        }
        t3[j].Set(t2)
      }
      for j := 0; j < m; j++ {
        r.At(i, j).Set(t3[j])
      }
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Outer product of two vectors. The result is stored in r.
func (r *DenseMatrix) Outer(a, b ConstVector) Matrix {
  n, m := r.Dims()
  if a.Dim() != n || b.Dim() != m {
    panic("matrix/vector dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).Mul(a.ConstAt(i), b.ConstAt(j))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Compute the Jacobian of f at x_. The result is stored in r.
func (r *DenseMatrix) Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
  n, m := r.Dims()
  x := x_.CloneVector()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  // reallocate matrix if dimensions do not match
  if r == nil || x.Dim() != m || y.Dim() != n {
     n = y.Dim()
     m = x.Dim()
    *r = *NullDenseMatrix(x_.ElementType(), n, m)
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).SetValue(y.At(i).GetDerivative(j))
    }
  }
  return r
}

// Compute the Hessian of f at x_. The result is stored in r.
func (r *DenseMatrix) Hessian(f func(Vector) Scalar, x_ Vector) Matrix {
  n, m := r.Dims()
  // reallocate matrix if dimensions do not match
  if r == nil || x_.Dim() != n || n != m {
     n = x_.Dim()
     m = x_.Dim()
    *r = *NullDenseMatrix(x_.ElementType(), n, m)
  }
  x := x_.CloneVector()
  x.Variables(2)
  // evaluate function
  y := f(x)
  // copy second derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).SetValue(y.GetHessian(i, j))
    }
  }
  return r
}
