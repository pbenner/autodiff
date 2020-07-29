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

package gramSchmidt

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  Q Matrix
  R Matrix
}

/* -------------------------------------------------------------------------- */

func gramSchmidt(a, q, r Matrix, t ScalarType, n, m int) (Matrix, Matrix, error) {

  v := a.CloneMatrix()
  s := NullScalar(t)

  for i := 0; i < m; i++ {
    // r_ii = ||v_i||
    r.At(i, i).Vnorm(v.ConstCol(i))
    for k := 0; k < n; k++ {
      q.At(k, i).Div(v.ConstAt(k, i), r.ConstAt(i, i))
    }
    for j := i+1; j < m; j++ {
      w := v.ConstCol(j)
      r.At(i, j).VdotV(q.ConstCol(i), w)
      for k := 0; k < n; k++ {
        s.Mul(r.ConstAt(i, j), q.ConstAt(k, i))
        v.At(k, j).Sub(w.ConstAt(k), s)
      }
    }
  }
  return q, r, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  var q Matrix
  var r Matrix

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case InSitu:
      q = a.Q
      r = a.R
    }
  }
  if q == nil {
    q = NullDenseMatrix(t, n, m)
  } else {
    if u, v := q.Dims(); u != n || v != m {
      return nil, nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if r == nil {
    r = NullDenseMatrix(t, n, m)
  } else {
    if u, v := r.Dims(); u != n || v != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  return gramSchmidt(a, q, r, t, n, m)
}
