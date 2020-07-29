/* Copyright (C) 2015-2020 Philipp Benner
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

/* -------------------------------------------------------------------------- */

type MatrixConstIterator interface {
  CloneConstIterator() MatrixConstIterator
  GetConst() ConstScalar
  Ok      () bool
  Next    ()
  Index   () (int, int)
}

type MatrixMagicIterator interface {
  CloneMagicIterator() MatrixMagicIterator
  GetConst() ConstScalar
  GetMagic() MagicScalar
  Ok      () bool
  Next    ()
  Index   () (int, int)
}

type MatrixConstJointIterator interface {
  CloneConstJointIterator() MatrixConstJointIterator
  GetConst() (ConstScalar, ConstScalar)
  Ok      () bool
  Next    ()
  Index   () (int, int)
}

type MatrixIterator interface {
  CloneIterator() MatrixIterator
  GetConst() ConstScalar
  Get     () Scalar
  Ok      () bool
  Next    ()
  Index   () (int, int)
}

type MatrixJointIterator interface {
  CloneJointIterator() MatrixJointIterator
  GetConst() (ConstScalar, ConstScalar)
  Get     () (Scalar, ConstScalar)
  Ok      () bool
  Next    ()
  Index   () (int, int)
}

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type constMatrix interface {
  CloneConstMatrix ()                           ConstMatrix
  Dims             ()                           (int, int)
  Equals           (ConstMatrix, float64)       bool
  Table            ()                           string
  Int8At           (i, j int)                   int8
  Int16At          (i, j int)                   int16
  Int32At          (i, j int)                   int32
  Int64At          (i, j int)                   int64
  IntAt            (i, j int)                   int
  Float32At        (i, j int)                   float32
  Float64At        (i, j int)                   float64
  ConstAt          (i, j int)                   ConstScalar
  ConstSlice       (rfrom, rto, cfrom, cto int) ConstMatrix
  ConstRow         (i int)                      ConstVector
  ConstCol         (j int)                      ConstVector
  ConstDiag        ()                           ConstVector
  AsConstVector    ()                           ConstVector
  ConstIterator    ()                           MatrixConstIterator
  ConstIteratorFrom(i, j int)                   MatrixConstIterator
  IsSymmetric      (float64)                    bool
  // private methods
  storageLocation  () uintptr
  // json
  json.Marshaler
}

type ConstMatrix interface {
  ConstScalarContainer
  constMatrix
}

type matrix interface {
  constMatrix
  CloneMatrix         ()                           Matrix
  // other methods
  At                  (i, j int)                   Scalar
  Reset               ()
  // basic methods
  Set                 (ConstMatrix)
  Row                 (i int)                      Vector
  Col                 (j int)                      Vector
  Diag                ()                           Vector
  T                   ()                           Matrix
  Tip                 ()
  Export              (string)                     error
  Slice               (rfrom, rto, cfrom, cto int) Matrix
  Swap                (int, int, int, int)
  SwapRows            (int, int) error
  SwapColumns         (int, int) error
  PermuteRows         ([]int) error
  PermuteColumns      ([]int) error
  SymmetricPermutation(pi []int) error
  SetIdentity         ()
  // returns all elements of the matrix as
  // a vector, the order is unspecified
  AsVector            ()                           Vector
  // iterators
  Iterator            ()                           MatrixIterator
  IteratorFrom        (i, j int)                   MatrixIterator
  JointIterator       (b ConstMatrix)              MatrixJointIterator
  // math operations
  MaddM(a,             b ConstMatrix)              Matrix
  MaddS(a ConstMatrix, b ConstScalar)              Matrix
  MsubM(a,             b ConstMatrix)              Matrix
  MsubS(a ConstMatrix, b ConstScalar)              Matrix
  MmulM(a,             b ConstMatrix)              Matrix
  MmulS(a ConstMatrix, b ConstScalar)              Matrix
  MdivM(a,             b ConstMatrix)              Matrix
  MdivS(a ConstMatrix, b ConstScalar)              Matrix
  MdotM(a,             b ConstMatrix)              Matrix
  Outer(a,             b ConstVector)              Matrix
  Jacobian(f func(ConstVector) ConstVector, x_ MagicVector) Matrix
  Hessian (f func(ConstVector) ConstScalar, x_ MagicVector) Matrix
}

type Matrix interface {
  ScalarContainer
  matrix
}

type MagicMatrix interface {
  MagicScalarContainer
  matrix
  CloneMagicMatrix    ()                           MagicMatrix
  MagicAt             (i, j int)                   MagicScalar
  MagicSlice          (rfrom, rto, cfrom, cto int) MagicMatrix
  ResetDerivatives    ()
  AsMagicVector       ()                           MagicVector
  MagicIterator       ()                           MatrixMagicIterator
  MagicIteratorFrom   (i, j int)                   MatrixMagicIterator
}

/* constructors
 * -------------------------------------------------------------------------- */

func NullDenseMatrix(t ScalarType, rows, cols int) Matrix {
  switch t {
  case Int8Type:
    return NullDenseInt8Matrix(rows, cols)
  case Int16Type:
    return NullDenseInt16Matrix(rows, cols)
  case Int32Type:
    return NullDenseInt32Matrix(rows, cols)
  case Int64Type:
    return NullDenseInt64Matrix(rows, cols)
  case IntType:
    return NullDenseIntMatrix(rows, cols)
  case Float32Type:
    return NullDenseFloat32Matrix(rows, cols)
  case Float64Type:
    return NullDenseFloat64Matrix(rows, cols)
  case Real32Type:
    return NullDenseReal32Matrix(rows, cols)
  case Real64Type:
    return NullDenseReal64Matrix(rows, cols)
  default:
    panic("unknown type")
  }
}

func AsDenseMatrix(t ScalarType, m ConstMatrix) Matrix {
  switch t {
  case Int8Type:
    return AsDenseInt8Matrix(m)
  case Int16Type:
    return AsDenseInt16Matrix(m)
  case Int32Type:
    return AsDenseInt32Matrix(m)
  case Int64Type:
    return AsDenseInt64Matrix(m)
  case IntType:
    return AsDenseIntMatrix(m)
  case Float32Type:
    return AsDenseFloat32Matrix(m)
  case Float64Type:
    return AsDenseFloat64Matrix(m)
  case Real32Type:
    return AsDenseReal32Matrix(m)
  case Real64Type:
    return AsDenseReal64Matrix(m)
  default:
    panic("unknown type")
  }
}

func NullDenseMagicMatrix(t ScalarType, rows, cols int) MagicMatrix {
  switch t {
  case Real32Type:
    return NullDenseReal32Matrix(rows, cols)
  case Real64Type:
    return NullDenseReal64Matrix(rows, cols)
  default:
    panic("unknown type")
  }
}

func AsDenseMagicMatrix(t ScalarType, m ConstMatrix) MagicMatrix {
  switch t {
  case Real32Type:
    return AsDenseReal32Matrix(m)
  case Real64Type:
    return AsDenseReal64Matrix(m)
  default:
    panic("unknown type")
  }
}

func NullSparseMatrix(t ScalarType, rows, cols int) Matrix {
  switch t {
  case Int8Type:
    return NullSparseInt8Matrix(rows, cols)
  case Int16Type:
    return NullSparseInt16Matrix(rows, cols)
  case Int32Type:
    return NullSparseInt32Matrix(rows, cols)
  case Int64Type:
    return NullSparseInt64Matrix(rows, cols)
  case IntType:
    return NullSparseIntMatrix(rows, cols)
  case Float32Type:
    return NullSparseFloat32Matrix(rows, cols)
  case Float64Type:
    return NullSparseFloat64Matrix(rows, cols)
  case Real32Type:
    return NullSparseReal32Matrix(rows, cols)
  case Real64Type:
    return NullSparseReal64Matrix(rows, cols)
  default:
    panic("unknown type")
  }
}

func AsSparseMatrix(t ScalarType, m ConstMatrix) Matrix {
  switch t {
  case Int8Type:
    return AsSparseInt8Matrix(m)
  case Int16Type:
    return AsSparseInt16Matrix(m)
  case Int32Type:
    return AsSparseInt32Matrix(m)
  case Int64Type:
    return AsSparseInt64Matrix(m)
  case IntType:
    return AsSparseIntMatrix(m)
  case Float32Type:
    return AsSparseFloat32Matrix(m)
  case Float64Type:
    return AsSparseFloat64Matrix(m)
  case Real32Type:
    return AsSparseReal32Matrix(m)
  case Real64Type:
    return AsSparseReal64Matrix(m)
  default:
    panic("unknown type")
  }
}

func NullSparseMagicMatrix(t ScalarType, rows, cols int) MagicMatrix {
  switch t {
  case Real32Type:
    return NullSparseReal32Matrix(rows, cols)
  case Real64Type:
    return NullSparseReal64Matrix(rows, cols)
  default:
    panic("unknown type")
  }
}

func AsSparseMagicMatrix(t ScalarType, m ConstMatrix) MagicMatrix {
  switch t {
  case Real32Type:
    return AsSparseReal32Matrix(m)
  case Real64Type:
    return AsSparseReal64Matrix(m)
  default:
    panic("unknown type")
  }
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func DenseIdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullDenseMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.At(i, i).SetFloat64(1.0)
  }
  return matrix
}

func SparseIdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullSparseMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.At(i, i).SetFloat64(1.0)
  }
  return matrix
}

func DenseMagicIdentityMatrix(t ScalarType, dim int) MagicMatrix {
  matrix := NullDenseMagicMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.At(i, i).SetFloat64(1.0)
  }
  return matrix
}

func SparseMagicIdentityMatrix(t ScalarType, dim int) MagicMatrix {
  matrix := NullSparseMagicMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.At(i, i).SetFloat64(1.0)
  }
  return matrix
}
