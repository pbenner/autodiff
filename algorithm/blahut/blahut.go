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

package blahut

/* -------------------------------------------------------------------------- */

import   "math"
import . "github.com/pbenner/autodiff"

/* initialization of data structures
 * -------------------------------------------------------------------------- */

func blahut_init_q(n, m int) Matrix {
  return NullDenseFloat64Matrix(m, n)
}

func blahut_init_r(n int) Vector {
  return NullDenseFloat64Vector(n)
}

/* naive Blahut implementation
 * -------------------------------------------------------------------------- */

func blahut_compute_q(channel Matrix, p Vector, q Matrix) {
  n, m := channel.Dims()
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      q.At(j, i).Mul(channel.ConstAt(i, j), p.ConstAt(i))
    }
    normalizeMatrixRow(q, j)
  }
}

func blahut_compute_r(channel, q Matrix, r Vector) {
  n, m := channel.Dims()
  t1 := NewScalar(q.ElementType(), 0.0)
  t2 := NewScalar(q.ElementType(), 0.0)
  for i := 0; i < n; i++ {
    r.At(i).SetFloat64(0.0)
    for j := 0; j < m; j++ {
      if !math.IsInf(math.Log(channel.ConstAt(i, j).GetFloat64()), -1) && // 0 log q = 0
         !math.IsInf(math.Log(      r.ConstAt(i   ).GetFloat64()),  1) {  // Inf + x = Inf
        r.At(i).Sub(r.ConstAt(i), t1.Mul(channel.ConstAt(i, j), t2.Log(q.ConstAt(j, i))))
      }
    }
    r.At(i).Exp(t1.Neg(r.ConstAt(i)))
  }
}

func blahut_compute_J(r Vector, J Scalar) {
  sum := NewScalar(r.ElementType(), 0.0)
  for i := 0; i < r.Dim(); i++ {
    sum.Add(sum, r.ConstAt(i))
  }
  J.Set(sum.Div(sum.Log(sum), ConstFloat64(math.Log(2.0))))
}

func blahut_compute_p(r Vector, lambda Scalar, p Vector) {
  t1 := NewScalar(r.ElementType(), 0.0)
  t2 := NewScalar(r.ElementType(), 0.0)
  for i := 0; i < p.Dim(); i++ {
    if math.IsInf(math.Log(p.ConstAt(i).GetFloat64()), -1) {
      p.At(i).Set(r.ConstAt(i))
    } else {
      // p[i] = p[i]^(1-lambda) * r[i]^lambda
      p.At(i).Mul(t1.Pow(p.At(i), t2.Sub(ConstFloat64(1.0), lambda)), t2.Pow(r.At(i), lambda))
    }
  }
  normalizeVector(p)
}

func blahut(channel Matrix, p_init Vector, steps int,
  hook func(Vector, Scalar) bool,
  lambda Scalar) Vector {

  n, m := channel.Dims()
  p := p_init.CloneVector()
  q := blahut_init_q(n, m)
  r := blahut_init_r(n)
  J := NewFloat64(0.0)

  for k := 0; k < steps; k++ {
    blahut_compute_q(channel, p, q)
    blahut_compute_r(channel, q, r)
    blahut_compute_J(r, J)
    blahut_compute_p(r, lambda, p)

    if hook != nil && hook(p, J) {
      break
    }
  }
  return p
}

/* main
 * -------------------------------------------------------------------------- */

func Run(channel Matrix, p_init Vector, steps int, args ...interface{}) Vector {
  // default values for optional parameters
  hook   := Hook  {nil}.Value
  lambda := Lambda{1.0}.Value

  // parse optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a.Value
    case Lambda:
      lambda = a.Value
    default:
      panic("blahut(): Invalid optional argument!")
    }
  }
  return blahut(channel, p_init, steps, hook, NewFloat64(lambda))
}
