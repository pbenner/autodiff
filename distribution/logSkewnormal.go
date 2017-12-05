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

import   "fmt"
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
    return nil, fmt.Errorf("NewLogSkewNormalDistribution(): Parameter dimensions do not match!")
  }
  t  := xi.ElementType()
  t1 := NewScalar(t, 0.0)
  // parameters for the multivariate normal
  // kappa = diag(s) omega diag(s)
  kappa := NullMatrix(t, n, n)
  for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
      kappa.At(i, j).Mul(t1.Mul(scale.At(i), scale.At(j)), omega.At(i,j))
    }
  }
  // parameters for the standard normal cdf
  mu    := NewVector(t,       []float64{0})
  sigma := NewMatrix(t, 1, 1, []float64{1})

  normal1, err := NewNormalDistribution(xi, kappa)
  if err != nil { return nil, err }
  normal2, err := NewNormalDistribution(mu, sigma)
  if err != nil { return nil, err }

  result := LogSkewNormalDistribution{*normal1, *normal2, omega, alpha, scale}

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *LogSkewNormalDistribution) Clone() *LogSkewNormalDistribution {
  return &LogSkewNormalDistribution{
    Normal1: *dist.Normal1.Clone(),
    Normal2: *dist.Normal2.Clone(),
    Omega  :  dist.Omega  .CloneMatrix(),
    Alpha  :  dist.Alpha  .CloneVector(),
    Scale  :  dist.Scale  .CloneVector() }
}

func (dist LogSkewNormalDistribution) Dim() int {
  return dist.Alpha.Dim()
}

func (dist *LogSkewNormalDistribution) ScalarType() ScalarType {
  return RealType
}

func (dist LogSkewNormalDistribution) LogPdf(r Scalar, x Vector) error {
  n := dist.Normal1.Dim()
  c := NewScalar(dist.ScalarType(), math.Log(2))
  y := NullVector(dist.ScalarType(), n)
  z := NullVector(dist.ScalarType(), n)
  t := NullVector(dist.ScalarType(), 1)
  s := NewScalar(dist.ScalarType(), 0.0)
  for i := 0; i < n; i++ {
    y.At(i).Log(s.Add(x.At(i), NewReal(1.0)))
    z.At(i).Div(s.Sub(y.At(i), dist.Normal1.Mu.At(i)), dist.Scale.At(i))
  }
  t.At(0).VdotV(dist.Alpha, z)
  // add the det Jacobian of the variable transform to the constant c
  for i := 0; i < n; i++ {
    c.Add(c, s.Neg(s.Log(y.At(i))))
  }

  r1 := r.CloneScalar()
  r2 := r.CloneScalar()

  if err := dist.Normal1.LogPdf(r1, x); err != nil {
    return err
  }
  if err := dist.Normal2.LogCdf(r2, t); err != nil {
    return err
  }
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
  p  = p.AppendVector(dist.Omega.ToVector())
  p  = p.AppendVector(dist.Alpha)
  p  = p.AppendVector(dist.Scale)
  return p
}

func (dist *LogSkewNormalDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  xi    := parameters.Slice(0*n+0*n*n,1*n+0*n*n)
  omega := parameters.Slice(1*n+0*n*n,1*n+1*n*n).ToMatrix(n, n)
  alpha := parameters.Slice(1*n+1*n*n,2*n+1*n*n)
  scale := parameters.Slice(2*n+1*n*n,3*n+1*n*n)
  if tmp, err := NewLogSkewNormalDistribution(xi, omega, alpha, scale); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
