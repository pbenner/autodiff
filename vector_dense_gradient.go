/* Copyright (C) 2019 Philipp Benner
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

/* vector type declaration
 * -------------------------------------------------------------------------- */

type DenseGradient struct {
  S Scalar
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) Dim() int {
  return obj.S.GetN()
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) ValueAt(i int) float64 {
  return obj.S.GetDerivative(i)
}

func (obj DenseGradient) ConstAt(i int) ConstScalar {
  return ConstReal(obj.S.GetDerivative(i))
}

func (obj DenseGradient) ConstSlice(i, j int) ConstVector {
  x := make([]float64, j-i)
  for k := i; k < j; k++ {
    x[k] = obj.S.GetDerivative(k)
  }
  return NewDenseConstRealVector(x)
}

func (obj DenseGradient) GetValues() []float64 {
  x := make([]float64, obj.Dim())
  for i := 0; i < obj.Dim(); i++ {
    x[i] = obj.S.GetDerivative(i)
  }
  return x
}

func (obj DenseGradient) ElementType() ScalarType {
  return BareRealType
}

/* -------------------------------------------------------------------------- */

func (obj DenseGradient) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i := 0; i < obj.Dim(); i++ {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(obj.ConstAt(i).String())
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (obj DenseGradient) Table() string {
  var buffer bytes.Buffer

  for i := 0; i < obj.Dim(); i++ {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(obj.ConstAt(i).String())
  }

  return buffer.String()
}

/* imlement ConstScalarContainer
 * -------------------------------------------------------------------------- */

func (obj DenseGradient) Reduce(f func(Scalar, ConstScalar) Scalar, r Scalar) Scalar {
  for i := 0; i < obj.Dim(); i++ {
    r = f(r, obj.ConstAt(i))
  }
  return r
}

/* -------------------------------------------------------------------------- */

// Test if elements in a equal elements in b.
func (a DenseGradient) Equals(b ConstVector, epsilon float64) bool {
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
