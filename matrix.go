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

import "encoding/json"

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type ConstMatrix interface {
  ConstScalarContainer
  Dims        ()                           (int, int)
  Equals      (ConstMatrix, float64)       bool
  Table       ()                           string
  ConstAt     (i, j int)                   ConstScalar
  ConstSlice  (rfrom, rto, cfrom, cto int) ConstMatrix
  ConstRow    (i int)                      ConstVector
  ConstCol    (j int)                      ConstVector
  ConstDiag   ()                           ConstVector
  // private methods
  storageLocation() uintptr
}

type Matrix interface {
  ScalarContainer
  // const methods
  Dims        ()                           (int, int)
  Equals      (ConstMatrix, float64)       bool
  Table       ()                           string
  ConstAt     (i, j int)                   ConstScalar
  ConstSlice  (rfrom, rto, cfrom, cto int) ConstMatrix
  ConstRow    (i int)                      ConstVector
  ConstCol    (j int)                      ConstVector
  ConstDiag   ()                           ConstVector
  // other methods
  At                  (i, j int)           Scalar
  Reset               ()
  ResetDerivatives    ()
  // basic methods
  CloneMatrix         ()                   Matrix
  Set                 (Matrix)
  Row                 (i int)              Vector
  Col                 (j int)              Vector
  Diag                ()                   Vector
  T                   ()                   Matrix
  Export              (string)             error
  Slice               (rfrom, rto, cfrom, cto int) Matrix
  Swap                (int, int, int, int)
  SwapRows            (int, int) error
  SwapColumns         (int, int) error
  PermuteRows         ([]int) error
  PermuteColumns      ([]int) error
  SymmetricPermutation(pi []int) error
  SetIdentity         ()
  IsSymmetric         (float64)            bool
  // returns all elements of the matrix as
  // a vector, the order is unspecified
  ToVector            ()                   Vector
  ToDenseMatrix       ()                  *DenseMatrix
  ToDenseVector       ()                   DenseVector
  // math operations
  MaddM(a,             b ConstMatrix)      Matrix
  MaddS(a ConstMatrix, b ConstScalar)      Matrix
  MsubM(a,             b ConstMatrix)      Matrix
  MsubS(a ConstMatrix, b ConstScalar)      Matrix
  MmulM(a,             b ConstMatrix)      Matrix
  MmulS(a ConstMatrix, b ConstScalar)      Matrix
  MdivM(a,             b ConstMatrix)      Matrix
  MdivS(a ConstMatrix, b ConstScalar)      Matrix
  MdotM(a,             b ConstMatrix)      Matrix
  Outer(a,             b ConstVector)      Matrix
  Jacobian(f func(ConstVector) ConstVector, x_ Vector) Matrix
  Hessian (f func(ConstVector) ConstScalar, x_ Vector) Matrix
  // json
  json.Marshaler
  // private methods
  storageLocation() uintptr
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewMatrix(t ScalarType, rows, cols int, values []float64) Matrix {
  if t == BareRealType {
    return NewDenseBareRealMatrix(rows, cols, values)
  } else {
    return NewDenseMatrix(t, rows, cols, values)
  }
}

func NullMatrix(t ScalarType, rows, cols int) Matrix {
  if t == BareRealType {
    return NullDenseBareRealMatrix(rows, cols)
  } else {
    return NullDenseMatrix(t, rows, cols)
  }
}

func NilMatrix(rows, cols int) Matrix {
  return NilDenseMatrix(rows, cols)
}
