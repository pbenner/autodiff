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

package blahut

/* -------------------------------------------------------------------------- */

import   "math"

/* naive Blahut implementation
 * -------------------------------------------------------------------------- */

func blahut_naive_compute_q(channel [][]float64, p []float64, q [][]float64) {
  n := len(channel)
  m := len(channel[0])
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      q[j][i] = channel[i][j]*p[i]
    }
    normalizeSlice(q[j])
  }
}

func blahut_naive_compute_r(channel, q [][]float64, r []float64) {
  n := len(channel)
  m := len(channel[0])
  for i := 0; i < n; i++ {
    r[i] = 0.0
    for j := 0; j < m; j++ {
      r[i] += channel[i][j]*math.Log(q[j][i])
    }
    r[i] = math.Exp(r[i])
  }
}

func blahut_naive_compute_J(r []float64, J *float64) {
  sum := 0.0
  for i, _ := range r {
    sum += r[i]
  }
  *J = math.Log(sum)/math.Log(2.0)
}

func blahut_naive_compute_p(r []float64, lambda float64, p []float64) {
  for i, _ := range p {
    p[i] = math.Pow(p[i], 1.0 - lambda)*math.Pow(r[i], lambda)
  }
  normalizeSlice(p)
}

func blahut_naive_init_p(p_init []float64) []float64 {
  p := make([]float64, len(p_init))
  copy(p, p_init)
  return p
}

func blahut_naive_init_q(n, m int) [][]float64 {
  q := make([][]float64, m)
  for j := 0; j < m; j++ {
    q[j] = make([]float64, n)
  }
  return q
}

func blahutNaive(channel [][]float64, p_init []float64, steps int,
  hook func([]float64, float64) bool,
  lambda float64) []float64 {

  n := len(channel)
  m := len(channel[0])
  p := blahut_naive_init_p(p_init)
  q := blahut_naive_init_q(n, m)
  r := make([]float64, n)
  J := 0.0

  for k := 0; k < steps; k++ {
    blahut_naive_compute_q(channel, p, q)
    blahut_naive_compute_r(channel, q, r)
    blahut_naive_compute_J(r, &J)
    blahut_naive_compute_p(r, lambda, p)

    if hook != nil && hook(p, J) {
      break
    }
  }
  return p
}

/* -------------------------------------------------------------------------- */

func RunNaive(channel [][]float64, p_init []float64, steps int, args ...interface{}) []float64 {
  // default values for optional parameters
  hook   := HookNaive{nil}.Value
  lambda := Lambda   {1.0}.Value

  // parse optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case HookNaive:
      hook = a.Value
    case Lambda:
      lambda = a.Value
    default:
      panic("blahut(): Invalid optional argument!")
    }
  }
  return blahutNaive(channel, p_init, steps, hook, lambda)
}
