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

// Test if elements in a equal elements in b.
func Vequal(a, b Vector) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for i := 0; i < a.Dim(); i++ {
    if !Equal(a.At(i), b.At(i)) {
      return false
    }
  }
  return true
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of two vectors. The result is stored in r.
func (r DenseVector) VaddV(a, b Vector) Vector {
  n := len(r)
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Add(a.At(i), b.At(i))
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r DenseVector) VaddS(a Vector, b Scalar) Vector {
  n := len(r)
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Add(a.At(i), b)
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of two vectors. The result is stored in r.
func (r DenseVector) VsubV(a, b Vector) Vector {
  n := len(r)
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Sub(a.At(i), b.At(i))
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substractor of a vector and a scalar. The result is stored in r.
func (r DenseVector) VsubS(a Vector, b Scalar) Vector {
  n := len(r)
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Sub(a.At(i), b)
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise multiplication of two vectors. The result is stored in r.
func (r DenseVector) VmulV(a Vector, b Vector) Vector {
  n := len(r)
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Mul(a.At(i), b.At(i))
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r DenseVector) VmulS(a Vector, s Scalar) Vector {
  n := len(r)
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Mul(a.At(i), s)
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of two vectors. The result is stored in r.
func (r DenseVector) VdivV(a Vector, b Vector) Vector {
  n := len(r)
  if a.Dim() != n || b.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Div(a.At(i), b.At(i))
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of a vector and a scalar. The result is stored in r.
func (r DenseVector) VdivS(a Vector, s Scalar) Vector {
  n := len(r)
  if a.Dim() != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < a.Dim(); i++ {
    r[i].Div(a.At(i), s)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func Vnorm(a Vector) Scalar {
  r := NullScalar(a.ElementType())
  r.Vnorm(a)
  return r
}
