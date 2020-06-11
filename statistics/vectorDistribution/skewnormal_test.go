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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestSkewNormalDistribution1(t *testing.T) {

	xi := NewVector(RealType, []float64{2, 3})
	omega := NewMatrix(RealType, 2, 2, []float64{16, 8, 8, 12})
	alpha := NewVector(RealType, []float64{2, 6})
	scale := NewVector(RealType, []float64{3, 4})

	normal, _ := NewSkewNormalDistribution(xi, omega, alpha, scale)

	x := NewVector(RealType, []float64{1, 2})
	y := NewReal(0.0)

	normal.LogPdf(y, x)

	if math.Abs(y.GetValue() - -1.025062e+01) > 1e-4 {
		t.Error("TestSkewNormalDistribution1 failed!")
	}
}

func TestSkewNormalDistribution2(t *testing.T) {

	xi := NewVector(RealType, []float64{4.906872e+00, 6.159821e+00})
	omega := NewMatrix(RealType, 2, 2, []float64{2.547269e+01, 3.171179e+00, 3.171179e+00, 2.547269e+01})
	alpha := NewVector(RealType, []float64{2.447269e+01, 2.447269e+01})
	scale := NewVector(RealType, []float64{3, 4})

	normal, _ := NewSkewNormalDistribution(xi, omega, alpha, scale)

	x := NewVector(RealType, []float64{2.75594661, 4.700348})
	y := NewReal(0.0)

	normal.LogPdf(y, x)

	if math.Abs(y.GetValue() - -3.615468e+02) > 1e-4 {
		t.Error("TestSkewNormalDistribution2 failed!")
	}
}

func TestSkewNormalFit1(t *testing.T) {
	// define the observed data
	x := NewMatrix(RealType, 100, 2, []float64{
		2.75594661, 4.700348,
		-0.55700646, 4.045626,
		6.70045102, 3.674610,
		0.42952381, 4.101212,
		6.59284684, 5.367460,
		4.32430179, 3.184627,
		0.64598990, 3.569413,
		6.31916051, 7.392589,
		2.04675683, 4.573011,
		7.82029543, 6.803890,
		3.09984252, 6.548937,
		5.06142204, 5.826152,
		9.06984778, 7.209964,
		-0.99438799, 4.361535,
		11.69984574, 13.438881,
		3.78897263, 4.100087,
		8.35420087, 13.960256,
		9.51603257, 4.896272,
		4.81957848, 5.570699,
		1.31990180, 4.888077,
		3.30109351, 3.307942,
		-0.01949095, 3.526769,
		-3.18444607, 8.465095,
		1.26720672, 6.005733,
		7.10484035, 10.055277,
		8.42091823, 9.113072,
		2.82901870, 4.027918,
		3.67989160, 3.855179,
		4.68795951, 6.820730,
		2.22447088, 5.715902,
		2.03225583, 5.818080,
		1.14186350, 5.493555,
		5.36202013, 9.655484,
		3.07788612, 5.199181,
		1.45947293, 4.724199,
		1.21795930, 3.225744,
		-0.28696571, 5.092004,
		3.90457320, 2.226318,
		4.58833263, 8.801928,
		2.04076021, 3.820850,
		6.16063043, 8.997081,
		2.73871857, 5.590378,
		9.57869898, 4.974781,
		2.78045704, 4.056074,
		8.95642718, 4.809547,
		6.61518318, 3.951484,
		0.98639482, 4.036594,
		-0.38829448, 4.185667,
		0.77141956, 4.346939,
		7.08872957, 4.931766,
		8.01350884, 3.798782,
		10.24393395, 4.244437,
		6.16115692, 6.343796,
		6.25459661, 3.130813,
		9.58238731, 8.162158,
		4.66971461, 10.677059,
		1.84788553, 8.660729,
		4.07800430, 9.379241,
		2.35532387, 10.701049,
		5.69698130, 5.498842,
		6.76664762, 5.074391,
		-1.83894184, 6.797524,
		7.67275951, 2.300542,
		0.28096349, 5.082953,
		0.35184918, 4.592556,
		9.57474386, 6.434783,
		0.23004606, 4.184062,
		9.53220682, 6.490756,
		6.55637517, 3.465605,
		9.07388670, 3.684909,
		5.83393443, 5.814029,
		5.37473026, 5.806414,
		4.16418916, 2.899504,
		9.72103570, 11.901487,
		9.04786447, 4.096438,
		0.51391140, 2.839885,
		4.51356571, 5.757988,
		2.83698717, 3.185261,
		3.13130469, 3.450239,
		5.18343076, 8.955890,
		3.69859097, 4.213072,
		5.26235420, 4.358703,
		4.68233028, 5.065302,
		3.94351170, 6.404511,
		4.71592292, 6.136997,
		2.25378555, 4.212412,
		4.71996431, 6.459409,
		0.84005735, 7.928305,
		2.04174016, 3.677648,
		7.46328759, 3.999967,
		1.50017507, 4.298077,
		8.34068452, 7.818200,
		2.68531625, 5.439663,
		6.04408035, 3.479537,
		2.44772785, 3.766574,
		9.17260944, 10.312474,
		5.94438534, 2.761207,
		3.91245029, 6.756676,
		2.10852175, 5.465862,
		9.47711202, 7.435752})
	y := NewReal(0.0)
	// number of data points
	n, _ := x.Dims()
	// define the (negative) likelihood function (here as a function of the
	// variables that we want to optimize)
	objective := func(variables Vector) (Scalar, error) {
		// create a new initial normal distribution
		xi := NullVector(RealType, 2)
		omega := NullMatrix(RealType, 2, 2)
		alpha := NullVector(RealType, 2)
		scale := NullVector(RealType, 2)
		// copy the variables
		xi.At(0).Set(variables.At(0))
		xi.At(1).Set(variables.At(1))
		omega.At(0, 0).Set(variables.At(2))
		omega.At(0, 1).Set(variables.At(3))
		omega.At(1, 0).Set(variables.At(3))
		omega.At(1, 1).Set(variables.At(5))
		alpha.At(0).Set(variables.At(6))
		alpha.At(1).Set(variables.At(7))
		scale.At(0).Set(variables.At(8))
		scale.At(1).Set(variables.At(9))
		normal, _ := NewSkewNormalDistribution(xi, omega, alpha, scale)
		result := NewScalar(RealType, 0.0)
		for i := 0; i < n; i++ {
			normal.LogPdf(y, x.Row(i))
			result.Add(result, y)
		}
		return Neg(Sub(result, NewReal(math.Log(float64(n))))), nil
	}
	// rprop hook
	// hook := func(gradient []float64, variables Vector, s Scalar) bool {
	//   fmt.Println(s)
	//   fmt.Println(variables)
	//   fmt.Println()
	//   return false
	// }
	// initial value
	v0 := NewVector(RealType, []float64{1, 1, 2, 1, 1, 2, 1, 1, 1, 1})
	// run rprop
	vn, _ := rprop.Run(objective, v0, 0.1, []float64{1.2, 0.2},
		//rprop.Hook{hook},
		rprop.Epsilon{1e-12})
	// check result
	if math.Abs(vn.At(0).GetValue()-2.583188e+00) > 1e-4 ||
		math.Abs(vn.At(1).GetValue()-2.920335e+00) > 1e-4 ||
		math.Abs(vn.At(2).GetValue()-2.948274e+00) > 1e-4 ||
		math.Abs(vn.At(3).GetValue()-1.646367e+00) > 1e-4 ||
		math.Abs(vn.At(4).GetValue()-1.0000000+00) > 1e-4 || // unused!
		math.Abs(vn.At(5).GetValue()-3.030302e+00) > 1e-4 ||
		math.Abs(vn.At(6).GetValue()-8.881898e-01) > 1e-4 ||
		math.Abs(vn.At(7).GetValue()-5.275696e+00) > 1e-4 ||
		math.Abs(vn.At(8).GetValue()-2.146501e+00) > 1e-4 ||
		math.Abs(vn.At(9).GetValue()-2.081685e+00) > 1e-4 {
		t.Error("TestSkewNormalFit2 failed!")
	}
}
