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

package matrixDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func init() {
	MatrixPdfRegistry["matrix:hierarchical hmm distribution"] = new(Hhmm)
	MatrixPdfRegistry["matrix:inverse wishart distribtion"] = new(InverseWishartDistribution)
	MatrixPdfRegistry["matrix:shape hmm distribution"] = new(ShapeHmm)
	MatrixPdfRegistry["matrix:hmm distribution"] = new(Hmm)
	MatrixPdfRegistry["matrix:mixture distribution"] = new(Mixture)
	MatrixPdfRegistry["matrix:vector id"] = new(VectorId)
	MatrixPdfRegistry["matrix:vector iid"] = new(VectorIid)
}
