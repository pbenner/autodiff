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

package generic

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type HmmDataRecord interface {
	// map index of the kth observation
	MapIndex(k int) int
	// get number of observations in this record
	GetN() int
	// evaluate component c at position k
	LogPdf(r Scalar, c, k int) error
}

/* -------------------------------------------------------------------------- */

type HmmDataSet interface {
	GetRecord(i int) HmmDataRecord
	// number of mapped observations
	GetNMapped() int
	// number of records in the data set
	GetNRecords() int
	// total number of observations
	GetN() int
}
