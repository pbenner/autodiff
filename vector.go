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

type VectorConstIterator interface {
  CloneConstIterator() VectorConstIterator
  GetConst() ConstScalar
  Ok      () bool
  Next    ()
  Index   () int
}

type VectorMagicIterator interface {
  CloneMagicIterator() VectorMagicIterator
  GetConst() ConstScalar
  Get     () Scalar
  GetMagic() MagicScalar
  Ok      () bool
  Next    ()
  Index   () int
}

type VectorConstJointIterator interface {
  CloneConstJointIterator() VectorConstJointIterator
  GetConst() (ConstScalar, ConstScalar)
  Ok      () bool
  Next    ()
  Index   () int
}

type VectorIterator interface {
  CloneIterator() VectorIterator
  GetConst() ConstScalar
  Get     () Scalar
  Ok      () bool
  Next    ()
  Index   () int
}

type VectorJointIterator interface {
  CloneJointIterator() VectorJointIterator
  GetConst() (ConstScalar, ConstScalar)
  Get     () (Scalar, ConstScalar)
  Ok      () bool
  Next    ()
  Index   () int
}

/* vector type declaration
 * -------------------------------------------------------------------------- */

type constVector interface {
  Dim               ()                     int
  Equals            (ConstVector, float64) bool
  Table             ()                     string
  Int8At            (int)                  int8
  Int16At           (int)                  int16
  Int32At           (int)                  int32
  Int64At           (int)                  int64
  IntAt             (int)                  int
  Float32At         (int)                  float32
  Float64At         (int)                  float64
  ConstAt           (int)                  ConstScalar
  ConstSlice        (i, j int)             ConstVector
  ConstIterator     ()                     VectorConstIterator
  ConstIteratorFrom (i int)                VectorConstIterator
  ConstJointIterator(ConstVector)          VectorConstJointIterator
  // json
  json.Marshaler
}

type ConstVector interface {
  ConstScalarContainer
  constVector
}

type vector interface {
  constVector
  CloneVector       ()                     Vector
  // const methods
  JointIterator     (ConstVector)          VectorJointIterator
  Iterator          ()                     VectorIterator
  IteratorFrom      (i int)                VectorIterator
  // other methods
  At                (int)                  Scalar
  Reset             ()
  // basic methods
  Set               (ConstVector)
  Slice             (i, j int)             Vector
  Export            (string)               error
  Permute           ([]int)                error
  ReverseOrder      ()
  Sort              (bool)
  AppendScalar      (...Scalar)            Vector
  AppendVector      (Vector)               Vector
  Swap              (i, j int)
  // type conversions
  AsMatrix          (n, m int)             Matrix
  // math operations
  VaddV(a,             b ConstVector)      Vector
  VaddS(a ConstVector, b ConstScalar)      Vector
  VsubV(a,             b ConstVector)      Vector
  VsubS(a ConstVector, b ConstScalar)      Vector
  VmulV(a,             b ConstVector)      Vector
  VmulS(a ConstVector, b ConstScalar)      Vector
  VdivV(a,             b ConstVector)      Vector
  VdivS(a ConstVector, b ConstScalar)      Vector
  MdotV(a ConstMatrix, b ConstVector)      Vector
  VdotM(a ConstVector, b ConstMatrix)      Vector
}

type Vector interface {
  ScalarContainer
  vector
}

type MagicVector interface {
  MagicScalarContainer
  vector
  CloneMagicVector ()               MagicVector
  ResetDerivatives ()
  MagicAt          (int)            MagicScalar
  MagicSlice       (i, j int)       MagicVector
  AppendMagicScalar(...MagicScalar) MagicVector
  AppendMagicVector(MagicVector)    MagicVector
  // iterators
  MagicIterator    ()               VectorMagicIterator
  MagicIteratorFrom(i int)          VectorMagicIterator
}

/* constructors
 * -------------------------------------------------------------------------- */

func NullDenseVector(t ScalarType, length int) Vector {
  switch t {
  case Int8Type:
    return NullDenseInt8Vector(length)
  case Int16Type:
    return NullDenseInt16Vector(length)
  case Int32Type:
    return NullDenseInt32Vector(length)
  case Int64Type:
    return NullDenseInt64Vector(length)
  case IntType:
    return NullDenseIntVector(length)
  case Float32Type:
    return NullDenseFloat32Vector(length)
  case Float64Type:
    return NullDenseFloat64Vector(length)
  case Real32Type:
    return NullDenseReal32Vector(length)
  case Real64Type:
    return NullDenseReal64Vector(length)
  default:
    panic("unknown type")
  }
}

func AsDenseVector(t ScalarType, v ConstVector) Vector {
  switch t {
  case Int8Type:
    return AsDenseInt8Vector(v)
  case Int16Type:
    return AsDenseInt16Vector(v)
  case Int32Type:
    return AsDenseInt32Vector(v)
  case Int64Type:
    return AsDenseInt64Vector(v)
  case IntType:
    return AsDenseIntVector(v)
  case Float32Type:
    return AsDenseFloat32Vector(v)
  case Float64Type:
    return AsDenseFloat64Vector(v)
  case Real32Type:
    return AsDenseReal32Vector(v)
  case Real64Type:
    return AsDenseReal64Vector(v)
  default:
    panic("unknown type")
  }
}

func NullDenseMagicVector(t ScalarType, length int) MagicVector {
  switch t {
  case Real32Type:
    return NullDenseReal32Vector(length)
  case Real64Type:
    return NullDenseReal64Vector(length)
  default:
    panic("unknown type")
  }
}

func AsDenseMagicVector(t ScalarType, v ConstVector) MagicVector {
  switch t {
  case Real32Type:
    return AsDenseReal32Vector(v)
  case Real64Type:
    return AsDenseReal64Vector(v)
  default:
    panic("unknown type")
  }
}

func NullSparseVector(t ScalarType, length int) Vector {
  switch t {
  case Int8Type:
    return NullSparseInt8Vector(length)
  case Int16Type:
    return NullSparseInt16Vector(length)
  case Int32Type:
    return NullSparseInt32Vector(length)
  case Int64Type:
    return NullSparseInt64Vector(length)
  case IntType:
    return NullSparseIntVector(length)
  case Float32Type:
    return NullSparseFloat32Vector(length)
  case Float64Type:
    return NullSparseFloat64Vector(length)
  case Real32Type:
    return NullSparseReal32Vector(length)
  case Real64Type:
    return NullSparseReal64Vector(length)
  default:
    panic("unknown type")
  }
}

func AsSparseVector(t ScalarType, v ConstVector) Vector {
  switch t {
  case Int8Type:
    return AsSparseInt8Vector(v)
  case Int16Type:
    return AsSparseInt16Vector(v)
  case Int32Type:
    return AsSparseInt32Vector(v)
  case Int64Type:
    return AsSparseInt64Vector(v)
  case IntType:
    return AsSparseIntVector(v)
  case Float32Type:
    return AsSparseFloat32Vector(v)
  case Float64Type:
    return AsSparseFloat64Vector(v)
  case Real32Type:
    return AsSparseReal32Vector(v)
  case Real64Type:
    return AsSparseReal64Vector(v)
  default:
    panic("unknown type")
  }
}

func NullSparseMagicVector(t ScalarType, length int) MagicVector {
  switch t {
  case Real32Type:
    return NullSparseReal32Vector(length)
  case Real64Type:
    return NullSparseReal64Vector(length)
  default:
    panic("unknown type")
  }
}

func AsSparseMagicVector(t ScalarType, v ConstVector) MagicVector {
  switch t {
  case Real32Type:
    return AsSparseReal32Vector(v)
  case Real64Type:
    return AsSparseReal64Vector(v)
  default:
    panic("unknown type")
  }
}
