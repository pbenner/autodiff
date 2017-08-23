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

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "errors"

/* -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) T() Matrix {
  return &DenseMatrix{
    values    :  matrix.values,
    rows      :  matrix.cols,
    cols      :  matrix.rows,
    transposed: !matrix.transposed,
    rowOffset :  matrix.colOffset,
    rowMax    :  matrix.colMax,
    colOffset :  matrix.rowOffset,
    colMax    :  matrix.rowMax,
    tmp1      :  matrix.tmp2,
    tmp2      :  matrix.tmp1 }
}

/* permutations
 * -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) SwapRows(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return errors.New("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < m; k++ {
    v1 := matrix.At(i, k)
    v2 := matrix.At(j, k)
    matrix.SetReferenceAt(v2, i, k)
    matrix.SetReferenceAt(v1, j, k)
  }
  return nil
}

func (matrix *DenseMatrix) SwapColumns(i, j int) error {
  n, m := matrix.Dims()
  if n != m {
    return errors.New("SymmetricPermutation(): matrix is not a square matrix")
  }
  for k := 0; k < n; k++ {
    v1 := matrix.At(k, i)
    v2 := matrix.At(k, j)
    matrix.SetReferenceAt(v2, k, i)
    matrix.SetReferenceAt(v1, k, j)
  }
  return nil
}

func (matrix *DenseMatrix) PermuteRows(pi []int) error {
  n, m := matrix.Dims()
  if n != m {
    return errors.New("SymmetricPermutation(): matrix is not a square matrix")
  }
  // permute matrix
  for i := 0; i < n; i++ {
    if pi[i] < 0 || pi[i] > n {
      return errors.New("SymmetricPermutation(): invalid permutation")
    }
    if i != pi[i] && pi[i] > i {
      matrix.SwapRows(i, pi[i])
    }
  }
  return nil
}

func (matrix *DenseMatrix) PermuteColumns(pi []int) error {
  n, m := matrix.Dims()
  if n != m {
    return errors.New("SymmetricPermutation(): matrix is not a square matrix")
  }
  // permute matrix
  for i := 0; i < m; i++ {
    if pi[i] < 0 || pi[i] > n {
      return errors.New("SymmetricPermutation(): invalid permutation")
    }
    if i != pi[i] && pi[i] > i {
      matrix.SwapColumns(i, pi[i])
    }
  }
  return nil
}

func (matrix *DenseMatrix) SymmetricPermutation(pi []int) error {
  n, m := matrix.Dims()
  if n != m {
    return errors.New("SymmetricPermutation(): matrix is not a square matrix")
  }
  for i := 0; i < n; i++ {
    if pi[i] < 0 || pi[i] > n {
      return errors.New("SymmetricPermutation(): invalid permutation")
    }
    if pi[i] > i {
      // permute rows
      matrix.SwapRows(i, pi[i])
      // permute colums
      matrix.SwapColumns(i, pi[i])
    }
  }
  return nil
}
