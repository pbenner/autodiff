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
import   "github.com/pbenner/autodiff/algorithm/determinant"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type NormalDistribution struct {
  Mu       Vector
  Sigma    Matrix
  SigmaInv Matrix
  SigmaDet Scalar
  logH     Scalar
}

/* -------------------------------------------------------------------------- */

func NewNormalDistribution(mu Vector, sigma Matrix) (*NormalDistribution, error) {

  n, m := sigma.Dims()
  t    := mu.ElementType()

  if n != m {
    panic("NewNormalDistribution(): sigma is not a square matrix!")
  }
  if n != len(mu) {
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
    logH    : h}

  return &result, nil

}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) Clone() *NormalDistribution {
  return &NormalDistribution{
    Mu      : dist.Mu      .Clone(),
    Sigma   : dist.Sigma   .Clone(),
    SigmaInv: dist.SigmaInv.Clone(),
    SigmaDet: dist.SigmaDet.Clone(),
    logH    : dist.logH    .Clone() }
}

func (dist *NormalDistribution) Dim() int {
  return len(dist.Mu)
}

func (dist *NormalDistribution) Mean() Vector {
  return dist.Mu.Clone()
}

func (dist *NormalDistribution) Variance() Vector {
  return dist.Sigma.Diag()
}

func (dist *NormalDistribution) LogH(x Vector) Scalar {
  return dist.logH
}

func (dist *NormalDistribution) LogPdf(x Vector) Scalar {
  t := x.ElementType()
  c := NewScalar(t, -0.5)
  // -1/2 [ p log(2pi) + log|Sigma| ]
  h := dist.LogH(x)
  // -1/2 (x-mu)^T Sigma^-1 (x-mu)
  y := VsubV(x, dist.Mu)
  s := Mul(c, VdotV(VdotM(y, dist.SigmaInv), y))

  return Add(h, s)
}

func (dist *NormalDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *NormalDistribution) LogCdf(x Vector) Scalar {
  if len(x) != 1 {
    panic("LogCdf(): not supported for more than one dimension.")
  }
  c := NewBareReal(2.0)
  y := Div(Sub(x[0], dist.Mu[0]), Sqrt(Mul(dist.Sigma.At(0,0), c)))
  r := Sub(LogErfc(Neg(y)), Log(c))

  // if computation of derivatives failed, return an approximation
  if r.GetOrder() >= 1 {
    for i := 0; i < r.GetN(); i++ {
      if math.IsNaN(r.GetDerivative(i)) && x[0].GetValue() < 0.0 {
        r.SetDerivative(i, -x[0].GetValue()*x[0].GetDerivative(i))
      }
    }
  }
  return r
}

func (dist *NormalDistribution) Cdf(x Vector) Scalar {
  return Exp(dist.LogCdf(x))
}


func (dist *NormalDistribution) EllipticCdf(x Vector) Scalar {
  d, _ := NewChiSquaredDistribution(2)
  // s = T(x) = (x-mu)^T Sigma^-1 (x-mu)
  y := VsubV(x, dist.Mu)
  s := VdotV(VdotM(y, dist.SigmaInv), y)
  // T(x) ~ chi^2_2
  return d.Cdf([]Scalar{s})
}

/* -------------------------------------------------------------------------- */

func (dist *NormalDistribution) GetParameters() Vector {
  p := dist.Mu
  p  = append(p, dist.Sigma.Vector()...)
  return p
}

func (dist *NormalDistribution) SetParameters(parameters Vector) error {
  n := dist.Dim()
  mu    := parameters[0:n]
  sigma := parameters[n:n+n*n].Matrix(n, n)
  if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}
