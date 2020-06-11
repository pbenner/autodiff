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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func init() {
	VectorPdfRegistry["vector:constrained hmm distribution"] = new(Chmm)
	VectorPdfRegistry["vector:hierarchical hmm distribution"] = new(Hhmm)
	VectorPdfRegistry["vector:hmm distribution"] = new(Hmm)
	VectorPdfRegistry["vector:mixture distribution"] = new(Mixture)
	VectorPdfRegistry["vector:normal distribtion"] = new(NormalDistribution)
	VectorPdfRegistry["vector:skew normal distribtion"] = new(SkewNormalDistribution)
	VectorPdfRegistry["vector:scalar id"] = new(ScalarId)
	VectorPdfRegistry["vector:scalar iid"] = new(ScalarIid)
	VectorPdfRegistry["vector:vector id"] = new(VectorId)
	VectorPdfRegistry["vector:vector iid"] = new(VectorIid)
}
