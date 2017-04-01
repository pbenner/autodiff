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

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type DensityFunction interface {
     Pdf(Vector) Scalar
  LogPdf(Vector) Scalar
  Dim() int
  GetParameters() Vector
  SetParameters(parameters Vector) error
}

type DistributionFunction interface {
     Cdf(Vector) Scalar
  LogCdf(Vector) Scalar
  Dim() int
  GetParameters() Vector
  SetParameters(parameters Vector) error
}

type Distribution interface {
     Pdf(Vector) Scalar
  LogPdf(Vector) Scalar
     Cdf(Vector) Scalar
  LogCdf(Vector) Scalar
  Dim() int
  GetParameters() Vector
  SetParameters(parameters Vector) error
}
