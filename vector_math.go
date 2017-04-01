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
  if len(a) != len(b) {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for i, _ := range (a) {
    if !Equal(a[i], b[i]) {
      return false
    }
  }
  return true
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of two vectors. The result is stored in r.
func (r Vector) VaddV(a, b Vector) Vector {
  n := len(r)
  if len(a) != n || len(b) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Add(a[i], b[i])
  }
  return r
}

// Element-wise addition of two vectors.
func VaddV(a, b Vector) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VaddV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of a vector and a scalar. The result is stored in r.
func (r Vector) VaddS(a Vector, b Scalar) Vector {
  n := len(r)
  if len(a) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Add(a[i], b)
  }
  return r
}

// Element-wise addition of a vector and a scalar.
func VaddS(a Vector, b Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VaddS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of two vectors. The result is stored in r.
func (r Vector) VsubV(a, b Vector) Vector {
  n := len(r)
  if len(a) != n || len(b) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Sub(a[i], b[i])
  }
  return r
}

// Element-wise substraction of two vectors.
func VsubV(a, b Vector) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VsubV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substractor of a vector and a scalar. The result is stored in r.
func (r Vector) VsubS(a Vector, b Scalar) Vector {
  n := len(r)
  if len(a) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Sub(a[i], b)
  }
  return r
}

// Element-wise substractor of a vector and a scalar.
func VsubS(a Vector, b Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VsubS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise multiplication of two vectors. The result is stored in r.
func (r Vector) VmulV(a Vector, b Vector) Vector {
  n := len(r)
  if len(a) != n || len(b) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Mul(a[i], b[i])
  }
  return r
}

// Element-wise multiplication of two vectors.
func VmulV(a Vector, b Vector) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VmulV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of a vector and a scalar. The result is stored in r.
func (r Vector) VmulS(a Vector, s Scalar) Vector {
  n := len(r)
  if len(a) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Mul(a[i], s)
  }
  return r
}

// Element-wise substraction of a vector and a scalar.
func VmulS(a Vector, s Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VmulS(a, s)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of two vectors. The result is stored in r.
func (r Vector) VdivV(a Vector, b Vector) Vector {
  n := len(r)
  if len(a) != n || len(b) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Div(a[i], b[i])
  }
  return r
}

// Element-wise division of two vectors.
func VdivV(a Vector, b Vector) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VdivV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of a vector and a scalar. The result is stored in r.
func (r Vector) VdivS(a Vector, s Scalar) Vector {
  n := len(r)
  if len(a) != n {
    panic("vector dimensions do not match")
  }
  for i := 0; i < len(a); i++ {
    r[i].Div(a[i], s)
  }
  return r
}

// Element-wise division of a vector and a scalar.
func VdivS(a Vector, s Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  r.VdivS(a, s)
  return r
}

/* -------------------------------------------------------------------------- */

func VdotV(a, b Vector) Scalar {
  r := NullScalar(a.ElementType())
  r.VdotV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

func Vnorm(a Vector) Scalar {
  r := NullScalar(a.ElementType())
  r.Vnorm(a)
  return r
}
