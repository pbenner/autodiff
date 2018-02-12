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
import "bytes"

/* -------------------------------------------------------------------------- */

type ConstRealVector []float64

/* constructors
 * -------------------------------------------------------------------------- */

func NewConstRealVector(v []float64) ConstRealVector {
  return ConstRealVector(v)
}

func NullConstRealVector(n int) ConstRealVector {
  return ConstRealVector(make([]float64, n))
}

/* -------------------------------------------------------------------------- */

func (v ConstRealVector) Dim() int {
  return len(v)
}

func (v ConstRealVector) ConstAt(i int) ConstScalar {
  return ConstReal(v[i])
}

func (v ConstRealVector) ConstSlice(i, j int) ConstVector {
  return v[i:j]
}

func (v ConstRealVector) GetValues() []float64 {
  return v
}

func (v ConstRealVector) ElementType() ScalarType {
  return RealType
}

/* -------------------------------------------------------------------------- */

func (v ConstRealVector) String() string {
  var buffer bytes.Buffer
  buffer.WriteString("[")
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(v.ConstAt(i).String())
  }
  buffer.WriteString("]")
  return buffer.String()
}

func (v ConstRealVector) Table() string {
  var buffer bytes.Buffer
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v.ConstAt(i).String())
  }
  return buffer.String()
}

/* math
 * -------------------------------------------------------------------------- */

func (a ConstRealVector) Equals(b ConstVector, epsilon float64) bool {
  if a.Dim() != b.Dim() {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for i := 0; i < a.Dim(); i++ {
    if !a.ConstAt(i).Equals(b.ConstAt(i), epsilon) {
      return false
    }
  }
  return true
}
