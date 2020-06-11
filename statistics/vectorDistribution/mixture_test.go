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
import "testing"
import "os"

import . "github.com/pbenner/autodiff/statistics"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {

	mu := NewVector(RealType, []float64{2, 3})
	sigma := NewMatrix(RealType, 2, 2, []float64{2, 1, 1, 2})

	normal, _ := NewNormalDistribution(mu, sigma)

	weights := NewVector(BareRealType, []float64{1.0, 2.0})

	mixture1, err := NewMixture(weights, []VectorPdf{normal, normal})
	if err != nil {
		t.Error(err)
		return
	}
	if err := ExportDistribution("mixture_test.json", mixture1); err != nil {
		t.Error(err)
	}
	mixture2, err := ImportVectorPdf("mixture_test.json", BareRealType)
	if err != nil {
		t.Error(err)
		return
	}

	normal1 := mixture1.Edist[0].(*NormalDistribution)
	normal2 := mixture2.(*Mixture).Edist[0].(*NormalDistribution)

	if Mnorm(MsubM(normal1.Sigma, normal2.Sigma)).GetValue() > 1e-8 {
		t.Error("test failed")
	}
	os.Remove("mixture_test.json")
}
