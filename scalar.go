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

import "fmt"
import "reflect"
import "encoding/json"

/* -------------------------------------------------------------------------- */

// this allows to idenfity the type of a scalar
type      ScalarType reflect.Type
type ConstScalarType reflect.Type

type ConstScalarState interface {
  // read access
  GetOrder        ()             int
  GetValue        ()             float64
  GetLogValue     ()             float64
  GetDerivative   (int)          float64
  GetHessian      (int, int)     float64
  GetN            ()             int
}

type ScalarState interface {
  ConstScalarState
  // allocate memory for derivatives of n variables
  Alloc           (int, int)
  // allocate enough memory for the derivatives of the given
  // variable(s) and copy the order
  AllocForOne     (ConstScalar)
  AllocForTwo     (ConstScalar, ConstScalar)
  // set value and derivatives to zero
  Reset           ()
  ResetDerivatives()
  // write access
  Set             (ConstScalar)
  setValue        (float64)
  SetValue        (float64)
  SetDerivative   (int, float64)
  SetHessian      (int, int, float64)
  SetVariable     (int, int, int) error
  // json
  json.Marshaler
}

type ConstScalar interface {
  ConstScalarState
  Equals      (ConstScalar, float64) bool
  Greater     (ConstScalar)          bool
  Smaller     (ConstScalar)          bool
  Sign        ()                     int
  // nice printing
  fmt.Stringer
}

type Scalar interface {
  ScalarState
  CloneScalar ()                Scalar
  // type reflections
  Type        ()                ScalarType
  ConvertType (ScalarType)      Scalar
  // some basic operations on scalars
  Equals      (ConstScalar, float64) bool
  Greater     (ConstScalar)          bool
  Smaller     (ConstScalar)          bool
  Sign        ()                int
  Min         (ConstScalar, ConstScalar)  Scalar
  Max         (ConstScalar, ConstScalar)  Scalar
  Abs         (ConstScalar)               Scalar
  Neg         (ConstScalar)               Scalar
  Add         (ConstScalar, ConstScalar)  Scalar
  Sub         (ConstScalar, ConstScalar)  Scalar
  Mul         (ConstScalar, ConstScalar)  Scalar
  Div         (ConstScalar, ConstScalar)  Scalar
  // add/substract the first two variables on log-scale,
  // take the third argument as a temporary variable
  LogAdd      (ConstScalar, ConstScalar, Scalar)  Scalar
  LogSub      (ConstScalar, ConstScalar, Scalar)  Scalar
  Pow         (ConstScalar, ConstScalar)          Scalar
  Sqrt        (ConstScalar)          Scalar
  Sin         (ConstScalar)          Scalar
  Sinh        (ConstScalar)          Scalar
  Cos         (ConstScalar)          Scalar
  Cosh        (ConstScalar)          Scalar
  Tan         (ConstScalar)          Scalar
  Tanh        (ConstScalar)          Scalar
  Exp         (ConstScalar)          Scalar
  Log         (ConstScalar)          Scalar
  Log1p       (ConstScalar)          Scalar
  Logistic    (ConstScalar)          Scalar
  Erf         (ConstScalar)          Scalar
  Erfc        (ConstScalar)          Scalar
  LogErfc     (ConstScalar)          Scalar
  Gamma       (ConstScalar)          Scalar
  Lgamma      (ConstScalar)          Scalar
  Mlgamma     (ConstScalar, int)     Scalar // multivariate log gamma
  GammaP      (float64, ConstScalar) Scalar // regularized lower incomplete gamma
  BesselI     (float64, ConstScalar) Scalar // modified bessel function of the first kind
  // vector operations
  SmoothMax   (x ConstVector, alpha ConstReal, t [2]Scalar) Scalar
  LogSmoothMax(x ConstVector, alpha ConstReal, t [3]Scalar) Scalar
  Vmean       (a    ConstVector)     Scalar
  VdotV       (a, b ConstVector)     Scalar
  Vnorm       (a    ConstVector)     Scalar
  Mnorm       (a    ConstMatrix)     Scalar
  Mtrace      (a    ConstMatrix)     Scalar
  // nice printing
  fmt.Stringer
}

/* keep a map of valid scalar implementations and a reference
 * to the constructors
 * -------------------------------------------------------------------------- */

type rtype map[ScalarType]func(float64) Scalar

// initialize empty registry
var scalarRegistry rtype = make(rtype)

// scalar types can be registered so that the constructors below can be used for
// all types
func RegisterScalar(t ScalarType, constructor func(float64) Scalar) {
  scalarRegistry[t] = constructor
}

/* constructors
 * -------------------------------------------------------------------------- */

func ScalarConstructor(t ScalarType) func(float64) Scalar {
  f, ok := scalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f
}

func NewScalar(t ScalarType, value float64) Scalar {
  f, ok := scalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f(value)
}

func NullScalar(t ScalarType) Scalar {
  f, ok := scalarRegistry[t]
  if !ok {
    panic(fmt.Sprintf("invalid scalar type `%v'", t))
  }
  return f(0.0)
}

/* -------------------------------------------------------------------------- */

func Variables(order int, reals ...Scalar) error {
  for i, _ := range reals {
    if err := reals[i].SetVariable(i, len(reals), order); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func CopyGradien(g Vector, x ConstScalar) error {
  n := x.GetN()
  if g.Dim() != n {
    return fmt.Errorf("vector has invalid length")
  }
  for i := 0; i < n; i++ {
    g.At(i).SetValue(x.GetDerivative(i))
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
      H.At(i, j).SetValue(x.GetHessian(i,j))
    }
  }
  return nil
}

func GetGradient(t ScalarType, x ConstScalar) Vector {
  n := x.GetN()
  g := NullVector(t, n)
  for i := 0; i < n; i++ {
    g.At(i).SetValue(x.GetDerivative(i))
  }
  return g
}

func GetHessian(t ScalarType, x ConstScalar) Matrix {
  n := x.GetN()
  H := NullMatrix(t, n, n)
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      H.At(i, j).SetValue(x.GetHessian(i,j))
    }
  }
  return H
}
