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

package eigensystem

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {
	a := NewMatrix(RealType, 6, 6, []float64{
		7, 3, 4, -11, -9, -2,
		-6, 4, -5, 7, 1, 12,
		-1, -9, 2, 2, 9, 1,
		-8, 0, -1, 5, 0, 8,
		-4, 3, -5, 7, 2, 10,
		6, 1, 4, -11, -7, -1})

	eigenvalues := []float64{5, 5, 4, 3, 1, 1}

	if r, v, err := Run(a, ComputeEigenvectors{false}); err != nil {
		t.Error("test failed")
	} else {
		for i := 0; i < 4; i++ {
			if math.Abs(r.At(i).GetValue()-eigenvalues[i]) > 1e-5 {
				t.Errorf("test failed for eigenvalue `%d'", i)
			}
		}
		if v != nil {
			t.Error("test failed")
		}
	}
}

func Test2(t *testing.T) {
	a := NewMatrix(RealType, 4, 4, []float64{
		1, 2, 3, 4,
		4, 4, 4, 4,
		0, 1, -1, 1,
		0, 0, 2, 3})

	eigenvalues := NewVector(RealType, []float64{
		6.741657e+00, 2.561553e+00, -1.561553e+00, -7.416574e-01})
	eigenvectors := NewMatrix(RealType, 4, 4, []float64{
		4.229518e-01, -6.818712e-02, 3.347805e-01, -7.125998e-01,
		8.951414e-01, -8.649304e-01, 1.055702e-01, 4.020316e-01,
		1.242021e-01, -1.064778e-01, -8.575579e-01, 5.070618e-01,
		6.638882e-02, 4.857041e-01, 3.759939e-01, -2.710359e-01})

	if e, v, err := Run(a); err != nil {
		t.Error(err)
	} else {
		if Vnorm(VsubV(eigenvalues, e)).GetValue() > 1e-4 {
			t.Error("test failed")
		}
		if Mnorm(MsubM(eigenvectors, v)).GetValue() > 1e-4 {
			t.Error("test failed")
		}
	}
}
