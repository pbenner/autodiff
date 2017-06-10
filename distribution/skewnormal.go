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
import   "errors"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type SkewNormalDistribution struct {
  Normal1 NormalDistribution
  Normal2 NormalDistribution
  Xi      Vector
  Omega   Matrix
  Alpha   Vector
  Scale   Vector
  l2      Scalar
  // state
  r1      Scalar
  r2      Scalar
  t       Vector
  z       Vector
}

/* -------------------------------------------------------------------------- */

// Azzalini, Adelchi, and Alessandra Dalla Valle. "The multivariate skew-normal
// distribution." Biometrika 83.4 (1996): 715-726.

func NewSkewNormalDistribution(xi Vector, omega Matrix, alpha Vector, scale Vector) (*SkewNormalDistribution, error) {
  t := xi.ElementType()
  // dimension
  n, m := omega.Dims()
  // check parameter dimensions
  if n != xi   .Dim()  ||
    (n != alpha.Dim()) ||
    (n != scale.Dim()) ||
    (n != m) {
    return nil, errors.New("NewSkewNormalDistribution(): Parameter dimensions do not match!")
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

  result := SkewNormalDistribution{
    Normal1: *normal1,
    Normal2: *normal2,
    Xi     : xi,
    Omega  : omega.CloneMatrix(),
    Alpha  : alpha.CloneVector(),
    Scale  : scale.CloneVector(),
    l2     : NewScalar(t, math.Log(2)),
    r1     : NewScalar(t, 0.0),
    r2     : NewScalar(t, 0.0),
    t      : NullVector(t, 1),
    z      : NullVector(t, n) }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *SkewNormalDistribution) Clone() *SkewNormalDistribution {
  return &SkewNormalDistribution{
    Normal1: *dist.Normal1.Clone(),
    Normal2: *dist.Normal2.Clone(),
    Xi     :  dist.Xi     .CloneVector(),
    Omega  :  dist.Omega  .CloneMatrix(),
    Alpha  :  dist.Alpha  .CloneVector(),
    Scale  :  dist.Scale  .CloneVector(),
    l2     :  dist.l2     .CloneScalar(),
    r1     :  dist.r1     .CloneScalar(),
    r2     :  dist.r2     .CloneScalar(),
    t      :  dist.t      .CloneVector(),
    z      :  dist.z      .CloneVector() }
}

func (dist *SkewNormalDistribution) Dim() int {
  return dist.Alpha.Dim()
}

func (dist *SkewNormalDistribution) LogPdf(r0 Scalar, x Vector) error {
  n := dist.Normal1.Dim()
  z := dist.z
  t := dist.t
  for i := 0; i < n; i++ {
    t.At(0).Sub(x.At(i), dist.Normal1.Mu.At(i))
    z.At(i).Div(t.At(0), dist.Scale.At(i))
  }
  t.At(0).VdotV(dist.Alpha, z)

  r1 := dist.r1
  r2 := dist.r2

  dist.Normal1.LogPdf(r1, x)
  dist.Normal2.LogCdf(r2, t)

  r0.Add(r1, r2)
  r0.Add(r0, dist.l2)

  return nil
}

func (dist *SkewNormalDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist SkewNormalDistribution) GetParameters() Vector {
  p := dist.Xi
  p  = p.Append(dist.Omega.DenseVector()...)
  p  = p.Append(dist.Alpha.DenseVector()...)
  p  = p.Append(dist.Scale.DenseVector()...)
  return p
}

func (dist *SkewNormalDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  xi    := parameters.Slice(0*n+0*n*n,1*n+0*n*n)
  omega := parameters.Slice(1*n+0*n*n,1*n+1*n*n).Matrix(n, n)
  alpha := parameters.Slice(1*n+1*n*n,2*n+1*n*n)
  scale := parameters.Slice(2*n+1*n*n,3*n+1*n*n)
  if tmp, err := NewSkewNormalDistribution(xi, omega, alpha, scale); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
