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

package msqrt

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "errors"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

// Denma-Beavers algorithm (not guaranteed to converge!)
// Other methods rely on the Schur decomposition, see:
// Higham, N.~J. (2008). Functions of Matrices: Theory and Computation;
// Society for Industrial and Applied Mathematics, Philadelphia, PA, USA.

func mSqrt(matrix Matrix) (Matrix, error) {
  n, _ := matrix.Dims()
  c  := NewScalar(matrix.ElementType(), 0.5)
  Y0 := matrix
  Z0 := IdentityMatrix(matrix.ElementType(), n)
  t1, err := matrixInverse.Run(Z0)
  if err != nil {
    return nil, err
  }
  t2, err := matrixInverse.Run(Y0)
  if err != nil {
    return nil, err
  }
  Y1 := MmulS(MaddM(Y0, t1), c)
  Z1 := MmulS(MaddM(Z0, t2), c)
  for Mnorm(MsubM(Y0, Y1)).GetValue() > 1e-8 {
    Y0 = Y1
    Z0 = Z1
    t1, err := matrixInverse.Run(Z0)
    if err != nil {
      return nil, err
    }
    t2, err := matrixInverse.Run(Y0)
    if err != nil {
      return nil, err
    }
    Y1 = MmulS(MaddM(Y0, t1), c)
    Z1 = MmulS(MaddM(Z0, t2), c)
  }
  return Y1, nil
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) (Matrix, error) {
  rows, cols := matrix.Dims()
  if rows != cols {
    return nil, errors.New("MSqrt(): Not a square matrix!")
  }
  if rows == 0 {
    return nil, errors.New("MSqrt(): Empty matrix!")
  }
  return mSqrt(matrix)
}
