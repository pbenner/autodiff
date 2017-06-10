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
import   "github.com/pbenner/autodiff/algorithm/determinant"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type NormalDistribution struct {
  Mu       Vector
  Sigma    Matrix
  SigmaInv Matrix
  SigmaDet Scalar
  logH     Scalar
  c1       Scalar
  c2       Scalar
  c3       Scalar
  // state
  t1       Vector
  t2       Vector
  t3       Scalar
}

/* -------------------------------------------------------------------------- */

func NewNormalDistribution(mu Vector, sigma Matrix) (*NormalDistribution, error) {

  n, m := sigma.Dims()
  t    := mu.ElementType()

  if n != m {
    panic("NewNormalDistribution(): sigma is not a square matrix!")
  }
  if n != mu.Dim() {
    panic("NewNormalDistribution(): dimensions of mu and sigma do not match!")
  }
  sigmaInv, err := matrixInverse.Run(sigma, matrixInverse.PositiveDefinite{true})
  if err != nil { return nil, err }
  sigmaDet, err := determinant  .Run(sigma, determinant  .PositiveDefinite{true})
  if err != nil { return nil, err }

  // -1/2 [ p log(2pi) + log|Sigma| ]
  c1 := NewScalar(t, -0.5)
  c2 := NewScalar(t, float64(n)*math.Log(2*math.Pi))
  h  := Mul(c1, Add(c2, Log(Abs(sigmaDet))))

  result := NormalDistribution{
    Mu      : mu,
    Sigma   : sigma,
    SigmaInv: sigmaInv,
    SigmaDet: sigmaDet,
    logH    : h,
    c1      : NewScalar(t, 1.0),
    c2      : NewScalar(t, 2.0),
    c3      : NewScalar(t, math.Log(2.0)),
    t1      : NullVector(t, n),
    t2      : NullVector(t, n),
    t3      : NullScalar(t) }

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) CloneScalar() *NormalDistribution {
  return &NormalDistribution{
    Mu      : dist.Mu      .CloneVector(),
    Sigma   : dist.Sigma   .CloneMatrix(),
    SigmaInv: dist.SigmaInv.CloneMatrix(),
    SigmaDet: dist.SigmaDet.CloneScalar(),
    logH    : dist.logH    .CloneScalar(),
    c1      : dist.c1      .CloneScalar(),
    c2      : dist.c2      .CloneScalar(),
    c3      : dist.c3      .CloneScalar(),
    t1      : dist.t1      .CloneVector(),
    t2      : dist.t2      .CloneVector(),
    t3      : dist.t3      .CloneScalar() }
}

func (dist *NormalDistribution) Dim() int {
  return dist.Mu.Dim()
}

func (dist *NormalDistribution) ScalarType() ScalarType {
  return dist.Mu.ElementType()
}

func (dist *NormalDistribution) Mean() Vector {
  return dist.Mu.CloneVector()
}

func (dist *NormalDistribution) Variance() Vector {
  return dist.Sigma.Diag()
}

func (dist *NormalDistribution) LogH(x Vector) Scalar {
  return dist.logH
}

func (dist *NormalDistribution) LogPdf(r Scalar, x Vector) error {
  if x.Dim() != dist.Dim() {
    return fmt.Errorf("input vector has invalid dimension")
  }
  // -1/2 [ p log(2pi) + log|Sigma| ]
  h := dist.LogH(x)
  // -1/2 (x-mu)^T Sigma^-1 (x-mu)
  y := dist.t1
  s := dist.t2
  y.VsubV(x, dist.Mu)
  s.VdotM(y, dist.SigmaInv)
  r.VdotV(s, y)
  r.Div(r, dist.c2)
  r.Sub(h, r)

  return nil
}

func (dist *NormalDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *NormalDistribution) LogCdf(r Scalar, x Vector) error {
  if x.Dim() != 1 {
    panic("LogCdf(): not supported for more than one dimension.")
  }
  t := dist.t3
  t.Mul(dist.c2, dist.Sigma.At(0,0))
  t.Sqrt(t)

  r.Sub(x.At(0), dist.Mu.At(0))
  r.Div(r, t)
  r.Neg(r)
  r.LogErfc(r)
  r.Sub(r, dist.c3)

  // if computation of derivatives failed, return an approximation
  if r.GetOrder() >= 1 {
    for i := 0; i < r.GetN(); i++ {
      if math.IsNaN(r.GetDerivative(i)) && x.At(0).GetValue() < 0.0 {
        r.SetDerivative(i, -x.At(0).GetValue()*x.At(0).GetDerivative(i))
      }
    }
  }
  return nil
}

func (dist *NormalDistribution) Cdf(r Scalar, x Vector) error {
  if err := dist.LogCdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *NormalDistribution) EllipticCdf(r Scalar, x Vector) error {
  d, _ := NewChiSquaredDistribution(2)
  y := dist.t1
  s := dist.t2
  t := dist.t3
  // t = T(x) = (x-mu)^T Sigma^-1 (x-mu)
  y.VsubV(x, dist.Mu)
  s.VdotM(y, dist.SigmaInv)
  t.VdotV(s, y)
  // T(x) ~ chi^2_2
  return d.Cdf(r, DenseVector{t})
}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) GetParameters() Vector {
  p := dist.Mu
  p  = p.Append(dist.Sigma.DenseVector()...)
  return p
}

func (dist *NormalDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  mu    := parameters.Slice(0,n)
  sigma := parameters.Slice(n,n+n*n).Matrix(n, n)
  if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
