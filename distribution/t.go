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

type TDistribution struct {
  Nu       Scalar
  Mu       Vector
  Sigma    Matrix
  SigmaInv Matrix
  SigmaDet Scalar
}

/* -------------------------------------------------------------------------- */

func NewTDistribution(nu Scalar, mu Vector, sigma Matrix) (*TDistribution, error) {

  n, m := sigma.Dims()

  if n != m {
    panic("NewTDistribution(): sigma is not a square matrix!")
  }
  if n != len(mu) {
    panic("NewTDistribution(): dimensions of mu and sigma do not match!")
  }
  sigmaInv, err := matrixInverse.Run(sigma, matrixInverse.PositiveDefinite{true})
  if err != nil { return nil, err }
  sigmaDet, err := determinant  .Run(sigma, determinant  .PositiveDefinite{true})
  if err != nil { return nil, err }

  result := TDistribution{nu, mu, sigma, sigmaInv, sigmaDet}

  return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *TDistribution) Clone() *TDistribution {
  return &TDistribution{
    dist.Nu.Clone(), dist.Mu.Clone(), dist.Sigma.Clone(),
    dist.SigmaInv.Clone(), dist.SigmaDet.Clone()}
}

func (dist *TDistribution) Dim() int {
  return len(dist.Mu)
}

func (dist *TDistribution) Mean() Vector {
  if dist.Nu.GetValue() <= 1.0 {
    panic("mean undefined for given parameters")
  }
  return dist.Mu.Clone()
}

func (dist *TDistribution) Variance() Vector {
  if dist.Nu.GetValue() <= 2.0 {
    panic("variance undefined for given parameters")
  }
  m := MmulS(dist.Sigma, Div(dist.Nu, Sub(dist.Nu, NewReal(2.0))))

  return m.Diag()
}

func (dist *TDistribution) LogPdf(x Vector) Scalar {
  t  := x.ElementType()
  d2 := NewScalar(t, float64(dist.Dim())/2.0)
  n2 := Div(dist.Nu, NewReal(2.0))
  // +log Gamma(nu/2 + d/2)
  c  := Lgamma(Add(n2, d2))
  // -log Gamma(nu/2)
  c  = Sub(c, Lgamma(n2))
  // -1/2 log |Sigma|
  c  = Sub(c, Div(Log(dist.SigmaDet), NewReal(2.0)))
  // -d/2 log nu*pi
  c  = Sub(c, Mul(d2, Log(Mul(dist.Nu, NewReal(math.Pi)))))
  //////////////////
  y := VsubV(x, dist.Mu)
  // 1/nu (x-mu)^T Sigma^-1 (x-mu)
  r := Add(NewReal(1.0), Div(VdotV(VdotM(y, dist.SigmaInv), y), dist.Nu))
  r  = Mul(Add(n2, d2), Log(r))

  return Sub(c, r)
}

func (dist *TDistribution) Pdf(x Vector) Scalar {
  return Exp(dist.LogPdf(x))
}

func (dist *TDistribution) LogCdf(x Vector) Scalar {
  return Log(dist.Cdf(x))
}

func (dist *TDistribution) Cdf(x Vector) Scalar {
  panic("not implemented")
}
