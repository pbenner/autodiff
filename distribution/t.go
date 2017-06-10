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
  c1       Scalar
  np       Scalar
  t1       Vector
  t2       Vector
  z        Scalar
}

/* -------------------------------------------------------------------------- */

func NewTDistribution(nu Scalar, mu Vector, sigma Matrix) (*TDistribution, error) {

  t := nu.Type()

  n, m := sigma.Dims()

  if n != m {
    panic("NewTDistribution(): sigma is not a square matrix!")
  }
  if n != mu.Dim() {
    panic("NewTDistribution(): dimensions of mu and sigma do not match!")
  }
  sigmaInv, err := matrixInverse.Run(sigma, matrixInverse.PositiveDefinite{true})
  if err != nil { return nil, err }
  sigmaDet, err := determinant  .Run(sigma, determinant  .PositiveDefinite{true})
  if err != nil { return nil, err }

  c1 := NewScalar(t, 1.0)
  c2 := NewScalar(t, 2.0)
  d2 := NewScalar(t, float64(n)/2.0)
  n2 := Div(nu, c2)
  np := Add(n2, d2)
  // +log Gamma(nu/2 + d/2)
  z  := Lgamma(np)
  // -log Gamma(nu/2)
  z.Sub(z, Lgamma(n2))
  // -1/2 log |Sigma|
  z.Sub(z, Div(Log(sigmaDet), c2))
  // -d/2 log nu*pi
  z.Sub(z, Mul(d2, Log(Mul(nu, NewReal(math.Pi)))))

  result := TDistribution{
    Nu      : nu.CloneScalar(),
    Mu      : mu.CloneVector(),
    Sigma   : sigma,
    SigmaInv: sigmaInv,
    SigmaDet: sigmaDet,
    c1      : c1,
    np      : np,
    t1      : NullVector(t, n),
    t2      : NullVector(t, n),
    z       : z }

  return &result, nil
}

/* -------------------------------------------------------------------------- */

func (dist *TDistribution) CloneScalar() *TDistribution {
  return &TDistribution{
    Nu      : dist.Nu      .CloneScalar(),
    Mu      : dist.Mu      .CloneVector(),
    Sigma   : dist.Sigma   .CloneMatrix(),
    SigmaInv: dist.SigmaInv.CloneMatrix(),
    SigmaDet: dist.SigmaDet.CloneScalar(),
    c1      : dist.c1      .CloneScalar(),
    np      : dist.np      .CloneScalar(),
    t1      : dist.t1      .CloneVector(),
    t2      : dist.t2      .CloneVector(),
    z       : dist.z       .CloneScalar() }
}

func (dist *TDistribution) Dim() int {
  return dist.Mu.Dim()
}

func (dist *TDistribution) Mean() Vector {
  if dist.Nu.GetValue() <= 1.0 {
    panic("mean undefined for given parameters")
  }
  return dist.Mu.CloneVector()
}

func (dist *TDistribution) Variance() Vector {
  if dist.Nu.GetValue() <= 2.0 {
    panic("variance undefined for given parameters")
  }
  m := MmulS(dist.Sigma, Div(dist.Nu, Sub(dist.Nu, NewReal(2.0))))

  return m.Diag()
}

func (dist *TDistribution) LogPdf(r Scalar, x Vector) error {
  y := dist.t1
  s := dist.t2
  // 1 + 1/nu (x-mu)^T Sigma^-1 (x-mu)
  y.VsubV(x, dist.Mu)
  s.VdotM(y, dist.SigmaInv)
  r.VdotV(s, y)
  r.Div(r, dist.Nu)
  r.Add(r, dist.c1)
  // log r^[(v+p)/2]
  r.Log(r)
  r.Mul(r, dist.np)
  r.Sub(dist.z, r)

  return nil
}

func (dist *TDistribution) Pdf(r Scalar, x Vector) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}
