/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2020 Philipp Benner
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
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
package autodiff
/* -------------------------------------------------------------------------- */
//import "fmt"
/* -------------------------------------------------------------------------- */
// True if matrix a equals b.
func (a *DenseReal32Matrix) Equals(b ConstMatrix, epsilon float64) bool {
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
func (a *DenseReal32Matrix) EQUALS(b *DenseReal32Matrix, epsilon float64) bool {
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("MEqual(): matrix dimensions do not match!")
  }
  for i := 0; i < n1; i++ {
    for j := 0; j < m1; j++ {
      if !a.AT(i, j).EQUALS(b.AT(i, j), epsilon) {
        return false
      }
    }
  }
  return true
}
/* -------------------------------------------------------------------------- */
// Element-wise addition of two matrices. The result is stored in r.
func (r *DenseReal32Matrix) MaddM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MADDM(a, b *DenseReal32Matrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).ADD(a.AT(i, j), b.AT(i, j))
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Add scalar b to all elements of a. The result is stored in r.
func (r *DenseReal32Matrix) MaddS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MADDS(a *DenseReal32Matrix, b *Real32) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).ADD(a.AT(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise substraction of two matrices. The result is stored in r.
func (r *DenseReal32Matrix) MsubM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MSUBM(a, b *DenseReal32Matrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).SUB(a.AT(i, j), b.AT(i, j))
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Substract b from all elements of a. The result is stored in r.
func (r *DenseReal32Matrix) MsubS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MSUBS(a *DenseReal32Matrix, b *Real32) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).SUB(a.AT(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise multiplication of two matrices. The result is stored in r.
func (r *DenseReal32Matrix) MmulM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MMULM(a, b *DenseReal32Matrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).MUL(a.AT(i, j), b.AT(i, j))
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Multiply all elements of a with b. The result is stored in r.
func (r *DenseReal32Matrix) MmulS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MMULS(a *DenseReal32Matrix, b *Real32) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).MUL(a.AT(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Element-wise division of two matrices. The result is stored in r.
func (r *DenseReal32Matrix) MdivM(a, b ConstMatrix) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MDIVM(a, b *DenseReal32Matrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).DIV(a.AT(i, j), b.AT(i, j))
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Divide all elements of a by b. The result is stored in r.
func (r *DenseReal32Matrix) MdivS(a ConstMatrix, b ConstScalar) Matrix {
  n, m := r.Dims()
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
func (r *DenseReal32Matrix) MDIVS(a *DenseReal32Matrix, b *Real32) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).DIV(a.AT(i, j), b)
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Matrix product of a and b. The result is stored in r.
func (r *DenseReal32Matrix) MdotM(a, b ConstMatrix) Matrix {
  n , m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m2 != m || m1 != n2 {
    panic("matrix dimensions do not match!")
  }
  t1 := NewReal32(0.0)
  t2 := NewReal32(0.0)
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
        r.At(i, j).Set(t3.At(i))
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
        r.At(i, j).Set(t3.At(j))
      }
    }
  }
  return r
}
func (r *DenseReal32Matrix) MDOTM(a, b *DenseReal32Matrix) Matrix {
  n , m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m2 != m || m1 != n2 {
    panic("matrix dimensions do not match!")
  }
  t1 := NewReal32(0.0)
  t2 := NewReal32(0.0)
  if r.storageLocation() == b.storageLocation() {
    t3 := r.tmp1[0:n]
    for j := 0; j < m; j++ {
      for i := 0; i < n; i++ {
        t2.Reset()
        for k := 0; k < m1; k++ {
          t1.MUL(a.AT(i, k), b.AT(k, j))
          t2.ADD(t2, t1)
        }
        t3[i].SET(t2)
      }
      for i := 0; i < n; i++ {
        r.AT(i, j).SET(t3.AT(i))
      }
    }
  } else {
    t3 := r.tmp2[0:m]
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        t2.Reset()
        for k := 0; k < m1; k++ {
          t1.MUL(a.AT(i, k), b.AT(k, j))
          t2.ADD(t2, t1)
        }
        t3[j].SET(t2)
      }
      for j := 0; j < m; j++ {
        r.AT(i, j).SET(t3.AT(j))
      }
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Outer product of two vectors. The result is stored in r.
func (r *DenseReal32Matrix) Outer(a, b ConstVector) Matrix {
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
func (r *DenseReal32Matrix) OUTER(a, b DenseReal32Vector) Matrix {
  n, m := r.Dims()
  if a.Dim() != n || b.Dim() != m {
    panic("matrix/vector dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.AT(i, j).MUL(a.AT(i), b.AT(j))
    }
  }
  return r
}
/* -------------------------------------------------------------------------- */
// Compute the Jacobian of f at x_. The result is stored in r.
func (r *DenseReal32Matrix) Jacobian(f func(ConstVector) ConstVector, x_ MagicVector) Matrix {
  n, m := r.Dims()
  x := x_.CloneMagicVector()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  // reallocate matrix if dimensions do not match
  if x.Dim() != m || y.Dim() != n {
    panic("invalid dimension")
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).SetFloat64(y.ConstAt(i).GetDerivative(j))
    }
  }
  return r
}
// Compute the Hessian of f at x_. The result is stored in r.
func (r *DenseReal32Matrix) Hessian(f func(ConstVector) ConstScalar, x_ MagicVector) Matrix {
  n, m := r.Dims()
  // reallocate matrix if dimensions do not match
  if x_.Dim() != n || n != m {
    panic("invalid dimension")
  }
  x := x_.CloneMagicVector()
  x.Variables(2)
  // evaluate function
  y := f(x)
  // copy second derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.At(i, j).SetFloat64(y.GetHessian(i, j))
    }
  }
  return r
}
