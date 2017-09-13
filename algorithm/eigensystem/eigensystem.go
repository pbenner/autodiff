/* Copyright (C) 2017 Philipp Benner
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

package eigensystem

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "sort"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/backSubstitution"
import   "github.com/pbenner/autodiff/algorithm/qrAlgorithm"

/* -------------------------------------------------------------------------- */

type Symmetric struct {
  Value bool
}

type InSitu struct {
  QrAlgorithm qrAlgorithm.InSitu
}

/* sort eigenvalues by absolute value
 * -------------------------------------------------------------------------- */

type sortEigenvaluesType DenseVector

func (v sortEigenvaluesType) Len() int           { return len(v) }
func (v sortEigenvaluesType) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v sortEigenvaluesType) Less(i, j int) bool {
  return math.Abs(v[i].GetValue()) < math.Abs(v[j].GetValue())
}

func sortEigenvalues(v DenseVector) {
  sort.Sort(sort.Reverse(sortEigenvaluesType(v)))
}

/* -------------------------------------------------------------------------- */

func getEigenvalues(eigenvalues DenseVector, h Matrix, sort bool) {
  n, _ := h.Dims()
  for i := 0; i < n-1; i++ {
    if h.At(i+1,i).GetValue() == 0.0 {
      // real eigenvalue
      eigenvalues[i].Set(h.At(i,i).CloneScalar())
    } else {
      c2 := BareReal(2.0)
      // complex eigenvalues, drop complex part
      h11 := h.At(i+0,i+0)
      h22 := h.At(i+1,i+1)
      eigenvalues[i+0].Add(h11, h22)
      eigenvalues[i+0].Div(eigenvalues[i+0], &c2)
      eigenvalues[i+1].Set(eigenvalues[i+0])
      i++
    }
  }
  if h.At(n-1,n-2).GetValue() == 0.0 {
    eigenvalues[n-1].Set(h.At(n-1, n-1))
  }
  if sort {
    sortEigenvalues(eigenvalues)
  }
}

func getEigenvector(eigenvector Vector, eigenvalue Scalar, h, u Matrix, b Vector, k int) {
  inSitu := backSubstitution.InSitu{}
  // substract eigenvalue from diagonal
  for i := 0; i < k; i++ {
    h.At(i,i).Sub(h.At(i,i), eigenvalue)
  }
  // copy u
  for i := 0; i < k; i++ {
    b.At(i).Set(h.At(i,k))
    b.At(i).Neg(b.At(i))
  }
  if k > 0 {
    inSitu.X = eigenvector.Slice(0,k)
    backSubstitution.Run(h.Slice(0,k,0,k), b.Slice(0,k), &inSitu)
  }
  eigenvector.At(k).SetValue(1.0)
  // add eigenvalue to diagonal
  for i := 0; i < k; i++ {
    h.At(i,i).Add(h.At(i,i), eigenvalue)
  }
  b.Set(eigenvector)
  eigenvector.MdotV(u, b)
  b.At(0).Vnorm(eigenvector)
  eigenvector.VdivS(eigenvector, b.At(0))
}

func getEigenvectors(eigenvectors *DenseMatrix, eigenvalues DenseVector, h, u Matrix, b Vector) {
  n := eigenvalues.Dim()

  for i := 0; i < n; i++ {
    getEigenvector(eigenvectors.Col(i), eigenvalues[i], h, u, b, i)
  }
}

func sortEigensystem(eigenvectors *DenseMatrix, eigenvalues DenseVector) {
  m := make(map[Scalar]int)
  // permutation
  p := make([]int, eigenvalues.Dim())
  for i := 0; i < eigenvalues.Dim(); i++ {
    m[eigenvalues.At(i)] = i
  }
  sortEigenvalues(eigenvalues)

  for i := 0; i < eigenvalues.Dim(); i++ {
    p[i] = m[eigenvalues.At(i)]
  }
  eigenvectors.PermuteColumns(p)
}

/* -------------------------------------------------------------------------- */

func Eigenvalues(a Matrix, args_... interface{}) (Vector, error) {
  // default values for optional arguments
  inSitu := &InSitu{}
  // arguments passed on to the qrAlgorithm
  var args []interface{}
  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case qrAlgorithm.ComputeU:
      // drop this option
    case *InSitu:
      inSitu = tmp
    default:
      args = append(args, arg)
    }
  }
  args = append(args, qrAlgorithm.ComputeU{true})
  args = append(args, &inSitu.QrAlgorithm)

  n, _      := a.Dims()
  t         := a.ElementType()
  h, _, err := qrAlgorithm.Run(a, args...)
  if err != nil {
    return nil, err
  }
  eigenvalues := NullDenseVector(t, n)

  getEigenvalues(eigenvalues, h, true)

  return eigenvalues, nil
}

func Eigensystem(a Matrix, args_ ...interface{}) (Vector, Matrix, error) {
  n, _ := a.Dims()
  t    := a.ElementType()
  // default values for optional arguments
  symmetric := false
  inSitu    := &InSitu{}
  // arguments passed on to the qrAlgorithm
  var args []interface{}
  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case Symmetric:
      symmetric = tmp.Value
    case qrAlgorithm.ComputeU:
      // drop this option
    case *InSitu:
      inSitu = tmp
    default:
      args = append(args, arg)
    }
  }
  args = append(args, qrAlgorithm.ComputeU{true})
  args = append(args, &inSitu.QrAlgorithm)
  h, u, err := qrAlgorithm.Run(a, args...)
  if err != nil {
    return nil, nil, err
  }

  if symmetric {
    eigenvalues  := h.Diag().CloneVector().ToDenseVector()
    eigenvectors := u.ToDenseMatrix()

    sortEigensystem(eigenvectors, eigenvalues)

    return eigenvalues, eigenvectors, nil
  } else {
    eigenvalues  := NullDenseVector(t, n)
    eigenvectors := NullDenseMatrix(t, n, n)

    getEigenvalues (eigenvalues, h, false)
    getEigenvectors(eigenvectors, eigenvalues, h, u, inSitu.QrAlgorithm.T4)
    sortEigensystem(eigenvectors, eigenvalues)

    return eigenvalues, eigenvectors, nil
  }
}
