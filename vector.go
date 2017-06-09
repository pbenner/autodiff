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

package autodiff

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type Vector interface {
  ScalarContainer
  At              (int)                Scalar
  SetReferenceAt  (Scalar, int)
  Reset           ()
  ResetDerivatives()
  // basic methods
  CloneVector     ()                   Vector
  Set             (Vector)
  Dim             ()                   int
  Table           ()                   string
  Export          (string)             error
  Permute         ([]int)              error
  ReverseOrder    ()
  SortVector      (bool)               Vector
  Slice           (i, j int)           Vector
  Append          (...Scalar)          Vector
  // type conversions
  Matrix          (n, m int)           Matrix
  DenseVector     ()                   DenseVector
  // math operations
  VaddV(a, b Vector) Vector
  VaddS(a Vector, b Scalar) Vector
  VsubV(a, b Vector) Vector
  VsubS(a Vector, b Scalar) Vector
  VmulV(a, b Vector) Vector
  VmulS(a Vector, b Scalar) Vector
  VdivV(a, b Vector) Vector
  VdivS(a Vector, b Scalar) Vector
  MdotV(a Matrix, b Vector) Vector
  VdotM(a Vector, b Matrix) Vector
}
