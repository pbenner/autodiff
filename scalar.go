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

import "fmt"
import "reflect"
import "encoding/json"

/* -------------------------------------------------------------------------- */

type ConstScalar interface {
  // clone methods
  CloneConstScalar()             ConstScalar
  // read access
  GetInt8           ()           int8
  GetInt16          ()           int16
  GetInt32          ()           int32
  GetInt64          ()           int64
  GetInt            ()           int
  GetFloat32        ()           float32
  GetFloat64        ()           float64
  // magic access
  GetOrder          ()           int
  GetDerivative     (int)        float64
  GetHessian        (int, int)   float64
  GetN              ()           int
  // type reflections
  Type              ()           ScalarType
  ConvertConstScalar(ScalarType) ConstScalar
  // simple math
  Equals            (ConstScalar, float64) bool
  Greater           (ConstScalar)          bool
  Smaller           (ConstScalar)          bool
  Sign              ()                     int
  // nice printing
  fmt.Stringer
  // json
  json.Marshaler
}

/* -------------------------------------------------------------------------- */

type Scalar interface {
  ConstScalar
  // clone methods
  CloneScalar  ()                                          Scalar
  // write access
  Reset        ()
  Set          (ConstScalar)
  SetInt8      (int8)
  SetInt16     (int16)
  SetInt32     (int32)
  SetInt64     (int64)
  SetInt       (int)
  SetFloat32   (float32)
  SetFloat64   (float64)
  setInt8      (int8)
  setInt16     (int16)
  setInt32     (int32)
  setInt64     (int64)
  setInt       (int)
  setFloat32   (float32)
  setFloat64   (float64)
  // type reflections
  ConvertScalar(ScalarType) Scalar
  // some basic operations on scalars
  Min          (ConstScalar, ConstScalar)             Scalar
  Max          (ConstScalar, ConstScalar)             Scalar
  Abs          (ConstScalar)                          Scalar
  Neg          (ConstScalar)                          Scalar
  Add          (ConstScalar, ConstScalar)             Scalar
  Sub          (ConstScalar, ConstScalar)             Scalar
  Mul          (ConstScalar, ConstScalar)             Scalar
  Div          (ConstScalar, ConstScalar)             Scalar
  // add/substract the first two variables on log-scale,
  // take the third argument as a temporary variable
  LogAdd       (ConstScalar, ConstScalar, Scalar)     Scalar
  LogSub       (ConstScalar, ConstScalar, Scalar)     Scalar
  Log1pExp     (ConstScalar)                          Scalar
  Sigmoid      (ConstScalar,              Scalar)     Scalar
  Pow          (ConstScalar, ConstScalar)             Scalar
  Sqrt         (ConstScalar)                          Scalar
  Sin          (ConstScalar)                          Scalar
  Sinh         (ConstScalar)                          Scalar
  Cos          (ConstScalar)                          Scalar
  Cosh         (ConstScalar)                          Scalar
  Tan          (ConstScalar)                          Scalar
  Tanh         (ConstScalar)                          Scalar
  Exp          (ConstScalar)                          Scalar
  Log          (ConstScalar)                          Scalar
  Log1p        (ConstScalar)                          Scalar
  Logistic     (ConstScalar)                          Scalar
  Erf          (ConstScalar)                          Scalar
  Erfc         (ConstScalar)                          Scalar
  LogErfc      (ConstScalar)                          Scalar
  Gamma        (ConstScalar)                          Scalar
  Lgamma       (ConstScalar)                          Scalar
  Mlgamma      (ConstScalar, int)                     Scalar // multivariate log gamma
  GammaP       (float64, ConstScalar)                 Scalar // regularized lower incomplete gamma
  BesselI      (float64, ConstScalar)                 Scalar // modified bessel function of the first kind
  // vector operations
  SmoothMax    (x ConstVector, alpha ConstFloat64, t [2]Scalar) Scalar
  LogSmoothMax (x ConstVector, alpha ConstFloat64, t [3]Scalar) Scalar
  Vmean        (a    ConstVector)                     Scalar
  VdotV        (a, b ConstVector)                     Scalar
  Vnorm        (a    ConstVector)                     Scalar
  Mnorm        (a    ConstMatrix)                     Scalar
  Mtrace       (a    ConstMatrix)                     Scalar
}

/* -------------------------------------------------------------------------- */

type MagicScalar interface {
  Scalar
  CloneMagicScalar() MagicScalar
  // magic write access
  ResetDerivatives()
  SetDerivative   (int, float64)
  SetHessian      (int, int, float64)
  SetVariable     (int, int, int) error
  // allocate memory for derivatives of n variables
  Alloc           (int, int)
  // allocate enough memory for the derivatives of the given
  // variable(s) and copy the order
  AllocForOne     (ConstScalar)
  AllocForTwo     (ConstScalar, ConstScalar)
  // type reflections
  ConvertMagicScalar(ScalarType) MagicScalar
}

/* -------------------------------------------------------------------------- */

// this allows to idenfity the type of a scalar
type ScalarType reflect.Type

/* keep a map of valid scalar implementations and a reference
 * to the constructors
 * -------------------------------------------------------------------------- */

type rtype1 map[ScalarType]func(float64)      Scalar
type rtype2 map[ScalarType]func(float64) MagicScalar
type rtype3 map[ScalarType]func(float64) ConstScalar

// initialize empty registry
var      scalarRegistry rtype1 = make(rtype1)
var magicScalarRegistry rtype2 = make(rtype2)
var constScalarRegistry rtype3 = make(rtype3)

// scalar types can be registered so that the constructors below can be used for
// all types
func RegisterConstScalar(t ScalarType, constructor func(float64) ConstScalar) {
  constScalarRegistry[t] = constructor
}
func RegisterMagicScalar(t ScalarType, constructor func(float64) MagicScalar) {
  magicScalarRegistry[t] = constructor
}
func RegisterScalar(t ScalarType, constructor func(float64) Scalar) {
       scalarRegistry[t] = constructor
}

/* scalar constructors
 * -------------------------------------------------------------------------- */

func NewScalar(t ScalarType, value float64) Scalar {
  f, ok := scalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f(value)
}

func NullScalar(t ScalarType) Scalar {
  return NewScalar(t, 0.0)
}

/* const scalar constructors
 * -------------------------------------------------------------------------- */

func NewConstScalar(t ScalarType, value float64) ConstScalar {
  f, ok := scalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f(value)
}

func NullConstScalar(t ScalarType) ConstScalar {
  return NewConstScalar(t, 0.0)
}

/* magic scalar constructors
 * -------------------------------------------------------------------------- */

func NewMagicScalar(t ScalarType, value float64) MagicScalar {
  f, ok := magicScalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f(value)
}

func NullMagicScalar(t ScalarType) MagicScalar {
  return NewMagicScalar(t, 0.0)
}

/* -------------------------------------------------------------------------- */

func Variables(order int, reals ...MagicScalar) error {
  for i, _ := range reals {
    if err := reals[i].SetVariable(i, len(reals), order); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func CopyGradient(g Vector, x ConstScalar) error {
  n := x.GetN()
  if g.Dim() != n {
    return fmt.Errorf("vector has invalid length")
  }
  for i := 0; i < n; i++ {
    g.At(i).SetFloat64(x.GetDerivative(i))
  }
  return nil
}

func CopyHessian(H Matrix, x ConstScalar) error {
  n := x.GetN()
  if n1, n2 := H.Dims(); n1 != n || n2 != n {
    return fmt.Errorf("matrix has invalid dimensions")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      H.At(i, j).SetFloat64(x.GetHessian(i,j))
    }
  }
  return nil
}

func GetGradient(t ScalarType, x ConstScalar) Vector {
  n := x.GetN()
  g := NullDenseVector(t, n)
  for i := 0; i < n; i++ {
    g.At(i).SetFloat64(x.GetDerivative(i))
  }
  return g
}

func GetHessian(t ScalarType, x ConstScalar) Matrix {
  n := x.GetN()
  H := NullDenseMatrix(t, n, n)
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      H.At(i, j).SetFloat64(x.GetHessian(i,j))
    }
  }
  return H
}
