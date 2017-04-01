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

package msqrtInv

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "errors"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

// Sherif, Nagwa. "On the computation of a matrix inverse square root."
// Computing 46.4 (1991): 295-305.

func mSqrtInv(matrix Matrix) (Matrix, error) {
  n, _ := matrix.Dims()
  c  := NewScalar(matrix.ElementType(), 2.0)
  A  := matrix
  I  := IdentityMatrix(matrix.ElementType(), n)
  X0 := IdentityMatrix(matrix.ElementType(), n)
  t, err := matrixInverse.Run(MaddM(I, MdotM(A, MdotM(X0, X0))))
  if err != nil {
    return nil, err
  }
  X1 := MmulS(MdotM(X0, t), c)
  for Mnorm(MsubM(X0, X1)).GetValue() > 1e-8 {
    X0 = X1
    t, err := matrixInverse.Run(MaddM(I, MdotM(A, MdotM(X0, X0))))
    if err != nil {
      return nil, err
    }
    X1 = MmulS(MdotM(X0, t), c)
  }
  return X1, nil
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) (Matrix, error) {
  rows, cols := matrix.Dims()
  if rows != cols {
    return nil, errors.New("MSqrtInv(): Not a square matrix!")
  }
  if rows == 0 {
    return nil, errors.New("MSqrtInv(): Empty matrix!")
  }
  return mSqrtInv(matrix)
}
