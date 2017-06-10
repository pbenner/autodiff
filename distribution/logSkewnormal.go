/* Copyright (C) 2016 Philipp Benner
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

package distribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type LogSkewNormalDistribution struct {
  Normal1 NormalDistribution
  Normal2 NormalDistribution
  Omega   Matrix
  Alpha   Vector
  Scale   Vector
}

/* -------------------------------------------------------------------------- */

func NewLogSkewNormalDistribution(xi Vector, omega Matrix, alpha, scale Vector) (*LogSkewNormalDistribution, error) {
  // dimension
  n, m := omega.Dims()
  // check parameter dimensions
  if n != xi   .Dim()  ||
    (n != alpha.Dim()) ||
    (n != scale.Dim()) ||
    (n != m) {
    panic("NewLogSkewNormalDistribution(): Parameter dimensions do not match!")
  }
  // parameters for the multivariate normal
  // kappa = diag(s) omega diag(s)
  kappa := NullMatrix(RealType, n, n)
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      kappa.At(i, j).Mul(Mul(scale.At(i), scale.At(j)), omega.At(i,j))
    }
  }
  // parameters for the standard normal cdf
  mu    := NewVector(RealType,       []float64{0})
  sigma := NewMatrix(RealType, 1, 1, []float64{1})

  normal1, err := NewNormalDistribution(xi, kappa)
  if err != nil { return nil, err }
  normal2, err := NewNormalDistribution(mu, sigma)
  if err != nil { return nil, err }

  result := LogSkewNormalDistribution{*normal1, *normal2, omega, alpha, scale}

  return &result, nil

}

/* -------------------------------------------------------------------------- */


func (dist *LogSkewNormalDistribution) CloneScalar() *LogSkewNormalDistribution {
  return &LogSkewNormalDistribution{
    Normal1: *dist.Normal1.CloneScalar(),
    Normal2: *dist.Normal2.CloneScalar(),
    Omega  :  dist.Omega  .CloneMatrix(),
    Alpha  :  dist.Alpha  .CloneVector(),
    Scale  :  dist.Scale  .CloneVector() }
}

func (dist LogSkewNormalDistribution) Dim() int {
  return dist.Alpha.Dim()
}

func (dist LogSkewNormalDistribution) LogPdf(r Scalar, x Vector) error {
  n := dist.Normal1.Dim()
  c := NewScalar(RealType, math.Log(2))
  y := NullDenseVector(RealType, n)
  z := NullDenseVector(RealType, n)
  t := NullDenseVector(RealType, 1)
  for i := 0; i < n; i++ {
    y[i] = Log(Add(x.At(i), NewReal(1.0)))
    z[i] = Div(Sub(y.At(i), dist.Normal1.Mu.At(i)), dist.Scale.At(i))
  }
  t[0] = VdotV(dist.Alpha, z)
  // add the det Jacobian of the variable transform to the constant c
  for i := 0; i < n; i++ {
    c = Add(c, Neg(Log(y[i])))
  }

  r1 := r.CloneScalar()
  r2 := r.CloneScalar()

  dist.Normal1.LogPdf(r1, x)
  dist.Normal2.LogCdf(r2, t)

  r.Add(r1, r2)
  r.Add(r, c)

  return nil
}

func (dist LogSkewNormalDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogSkewNormalDistribution) GetParameters() Vector {
  p := dist.Normal1.Mu
  p  = p.Append(dist.Omega.DenseVector()...)
  p  = p.Append(dist.Alpha.DenseVector()...)
  p  = p.Append(dist.Scale.DenseVector()...)
  return p
}

func (dist *LogSkewNormalDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  xi    := parameters.Slice(0*n+0*n*n,1*n+0*n*n)
  omega := parameters.Slice(1*n+0*n*n,1*n+1*n*n).Matrix(n, n)
  alpha := parameters.Slice(1*n+1*n*n,2*n+1*n*n)
  scale := parameters.Slice(2*n+1*n*n,3*n+1*n*n)
  if tmp, err := NewLogSkewNormalDistribution(xi, omega, alpha, scale); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
